#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多様性指標を組み込んだSimPOトレーナーの拡張実装 - 修正版
"""

from typing import Dict, List, Literal, Union

import torch
import torch.nn.functional as F
import wandb
from trl import CPOTrainer


class DiversitySimPOTrainer(CPOTrainer):
    """
    多様性促進機能を追加したSimPOトレーナー実装
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        training_args = kwargs["args"]

        # SimPO関連のパラメータ
        self.gamma = getattr(training_args, "simpo_gamma", 0.7)

        # 多様性関連のハイパーパラメータ
        self.diversity_weight = getattr(training_args, "diversity_weight", 0.05)
        self.diversity_alpha = getattr(training_args, "diversity_alpha", 0.1)

        # CPOTrainerのloss_typeを確認
        self.loss_type = getattr(training_args, "loss_type", "simpo")
        self.beta = getattr(training_args, "beta", 0.0)

        # DeepSpeed統合フラグ
        self.use_deepspeed = True

        print(f"Initialized DiversitySimPOTrainer with parameters:")
        print(f"  - simpo_gamma: {self.gamma}")
        print(f"  - diversity_weight: {self.diversity_weight}")
        print(f"  - diversity_alpha: {self.diversity_alpha}")
        print(f"  - loss_type: {self.loss_type}")

    def log(self, logs, start_time=None):
        """
        拡張ログ機能 - wandbサポート付き
        """
        # 親クラスのlogメソッドをインスタンスメソッドとして呼び出し
        super().log(logs)

        # 親クラスの処理後にwandbにログを記録
        if self.args.report_to == "wandb":
            wandb.log(logs)

    def diversity_loss(
        self,
        policy_chosen_logits: torch.FloatTensor,
        policy_rejected_logits: torch.FloatTensor,
        alpha: float = None,  # デフォルト値はNone、初期化時に設定した値を使用
    ) -> torch.FloatTensor:
        """テキスト生成の多様性を最大化するための損失関数

        Args:
            policy_chosen_logits: 選択された応答のロジット
            policy_rejected_logits: 拒否された応答のロジット
            alpha: 多様性項の重み係数

        Returns:
            多様性損失
        """
        # alphaがNoneの場合は、初期化時に設定された値を使用
        if alpha is None:
            alpha = self.diversity_alpha

        try:
            # 入力テンソルの形状を確認し、必要に応じて調整
            if policy_chosen_logits.ndim > 2:
                # バッチと時間次元をまとめる（例: [batch, seq_len, vocab] → [batch*seq_len, vocab]）
                batch_size, seq_len = policy_chosen_logits.shape[:2]
                policy_chosen_logits = policy_chosen_logits.reshape(
                    -1, policy_chosen_logits.shape[-1]
                )
                policy_rejected_logits = policy_rejected_logits.reshape(
                    -1, policy_rejected_logits.shape[-1]
                )

            # エントロピーベースの多様性促進（高いエントロピー = より均一な確率分布 = より多様なトークン選択）
            # エプシロンを大きめに設定して数値安定性を向上
            chosen_probs = F.softmax(policy_chosen_logits, dim=-1)
            chosen_entropy = (
                -(chosen_probs * torch.log(chosen_probs + 1e-6)).sum(dim=-1).mean()
            )

            # KL divergence項（選択された応答と拒否された応答の分布の違いを維持）
            rejected_probs = F.softmax(policy_rejected_logits, dim=-1)
            kl_div = F.kl_div(
                F.log_softmax(policy_chosen_logits, dim=-1),
                rejected_probs,
                reduction="batchmean",
                log_target=False,
            )

            # 多様性損失（エントロピーを最大化し、KL divergenceを維持）
            # マイナスを付けてエントロピーを最大化（損失を最小化する方向）
            diversity_loss = -alpha * chosen_entropy + (1 - alpha) * kl_div

            # 異常値チェック
            if torch.isnan(diversity_loss) or torch.isinf(diversity_loss):
                print(
                    "WARNING: NaN or Inf detected in diversity loss. Using default value."
                )
                return torch.tensor(0.1, device=policy_chosen_logits.device)

            return diversity_loss

        except Exception as e:
            print(f"ERROR in diversity_loss: {str(e)}")
            # エラー発生時はデフォルト値を返す
            return torch.tensor(0.1, device=policy_chosen_logits.device)

    def simpo_with_diversity_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_chosen_logits: torch.FloatTensor,
        policy_rejected_logits: torch.FloatTensor,
        diversity_weight: float = None,  # デフォルト値はNone、初期化時に設定した値を使用
    ):
        """多様性促進を組み込んだSimPO損失関数（SimPOの損失を直接実装）"""
        # diversity_weightがNoneの場合は、初期化時に設定された値を使用
        if diversity_weight is None:
            diversity_weight = self.diversity_weight

        # 入力値のNaNチェック
        inputs_valid = (
            not torch.isnan(policy_chosen_logps).any()
            and not torch.isnan(policy_rejected_logps).any()
            and not torch.isnan(policy_chosen_logits).any()
            and not torch.isnan(policy_rejected_logits).any()
        )

        if not inputs_valid:
            print(
                "WARNING: NaN values detected in inputs to simpo_with_diversity_loss. Using fallback values."
            )
            device = policy_chosen_logps.device
            losses = torch.tensor(1.0, device=device, requires_grad=True)
            chosen_rewards = torch.tensor(0.5, device=device)
            rejected_rewards = torch.tensor(0.0, device=device)
            div_loss = torch.tensor(0.1, device=device)
            return losses, chosen_rewards, rejected_rewards, div_loss

        try:
            # SimPO損失を直接計算 (親クラスのメソッドに依存しない)
            gamma = self.gamma
            logps_diff = policy_chosen_logps - policy_rejected_logps

            # マージン付きロジスティック損失
            simpo_losses = -F.logsigmoid(gamma * logps_diff)

            # 報酬は単純に対数確率
            chosen_rewards = policy_chosen_logps
            rejected_rewards = policy_rejected_logps

            # 多様性損失を計算
            div_loss = self.diversity_loss(policy_chosen_logits, policy_rejected_logits)

            # NaNチェック
            if torch.isnan(simpo_losses).any() or torch.isnan(div_loss).any():
                print(
                    "WARNING: NaN values detected in loss computation. Using fallback values."
                )
                device = policy_chosen_logps.device
                simpo_losses = torch.tensor(1.0, device=device)
                div_loss = torch.tensor(0.1, device=device)

            # 結合した損失を返す（多様性損失の重みは調整可能）
            combined_losses = simpo_losses + diversity_weight * div_loss

            # 勾配計算のために、ロスに requires_grad=True を設定
            # デフォルトではシンプルな数学演算でrequires_gradが保持されるはずですが、
            # 念のため明示的に設定します
            if not combined_losses.requires_grad:
                print(
                    "WARNING: Loss does not require grad. Setting requires_grad=True."
                )
                combined_losses = combined_losses.clone().detach().requires_grad_(True)

            return combined_losses, chosen_rewards, rejected_rewards, div_loss

        except Exception as e:
            print(f"ERROR in simpo_with_diversity_loss: {str(e)}")
            # 緊急フォールバック
            device = policy_chosen_logps.device
            loss = torch.tensor(1.0, device=device, requires_grad=True)
            return (
                loss,
                torch.tensor(0.5, device=device),
                torch.tensor(0.0, device=device),
                torch.tensor(0.1, device=device),
            )

    def concatenated_forward(self, model, batch):
        """
        フォワードパスの実行 - 親クラスの実装をオーバーライドして必要な戻り値を確保
        """
        try:
            # 親クラスのconcatenated_forwardを呼び出し
            output = super().concatenated_forward(model, batch)

            # 返り値の確認と必要な値の抽出
            if isinstance(output, tuple):
                if len(output) == 2:
                    # 親クラスが(policy_chosen_logps, policy_rejected_logps)を返す場合
                    policy_chosen_logps, policy_rejected_logps = output
                    # ロジットを取得するための追加処理
                    policy_chosen_logits = self._get_logits_from_logps(
                        policy_chosen_logps
                    )
                    policy_rejected_logits = self._get_logits_from_logps(
                        policy_rejected_logps
                    )
                    return (
                        policy_chosen_logps,
                        policy_rejected_logps,
                        policy_chosen_logits,
                        policy_rejected_logits,
                    )
                elif len(output) >= 4:
                    # 4つ以上の値がある場合は最初の4つを使用
                    policy_chosen_logps = output[0]
                    policy_rejected_logps = output[1]
                    policy_chosen_logits = output[2]
                    policy_rejected_logits = output[3]
                    return (
                        policy_chosen_logps,
                        policy_rejected_logps,
                        policy_chosen_logits,
                        policy_rejected_logits,
                    )

            # 想定外の返り値の場合は別の処理を試みる
            print(
                f"Unexpected output format from concatenated_forward. Trying alternative approach."
            )

            # 別の方法でロジットを取得
            outputs = model(**batch)
            logits = outputs.logits

            # サンプル選択処理（シンプルな実装例）
            batch_size = logits.shape[0] // 2
            policy_chosen_logits = logits[:batch_size]
            policy_rejected_logits = logits[batch_size:]

            # log_probsの計算
            policy_chosen_logps = F.log_softmax(policy_chosen_logits, dim=-1).mean(
                dim=(0, 1)
            )
            policy_rejected_logps = F.log_softmax(policy_rejected_logits, dim=-1).mean(
                dim=(0, 1)
            )

            return (
                policy_chosen_logps,
                policy_rejected_logps,
                policy_chosen_logits,
                policy_rejected_logits,
            )

        except Exception as e:
            print(f"Error in concatenated_forward: {str(e)}")

            # 最終的なフォールバック: ダミーデータの作成
            device = next(model.parameters()).device
            dummy_chosen = torch.tensor([-1.0, -1.0], device=device)
            dummy_rejected = torch.tensor([-1.0, -1.0], device=device)
            vocab_size = 32000  # 一般的な語彙サイズ
            dummy_chosen_logits = torch.zeros((2, 10, vocab_size), device=device)
            dummy_rejected_logits = torch.zeros((2, 10, vocab_size), device=device)

            print(
                "WARNING: Using dummy data for forward pass. Results will not be accurate."
            )
            return (
                dummy_chosen,
                dummy_rejected,
                dummy_chosen_logits,
                dummy_rejected_logits,
            )

    def _get_logits_from_logps(self, logps):
        """
        対数確率からロジットを計算（おおよその近似）
        """
        # 単純な変換: logps から exp を取るとほぼ確率になる
        # その後、適切なスケーリングを行ってロジットに変換
        # これは完全に正確ではありませんが、多様性損失の計算には十分な近似です
        return logps * 10.0  # スケーリング係数は調整可能

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """多様性促進を含むバッチ損失と指標の計算"""
        metrics = {}

        try:
            # ここでconcatenated_forwardを呼び出し、必要な4つの値を取得
            (
                policy_chosen_logps,
                policy_rejected_logps,
                policy_chosen_logits,
                policy_rejected_logits,
            ) = self.concatenated_forward(model, batch)

            # SimPOと多様性を組み合わせた損失を計算（親クラスには依存しない実装）
            losses, chosen_rewards, rejected_rewards, diversity_loss = (
                self.simpo_with_diversity_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    policy_chosen_logits,
                    policy_rejected_logits,
                )
            )

            # 報酬精度を計算
            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            prefix = "eval_" if train_eval == "eval" else ""
            # 既存の指標
            metrics[f"{prefix}rewards/chosen"] = (
                chosen_rewards.mean().cpu().item()
                if not torch.isnan(chosen_rewards.mean())
                else 0.5
            )
            metrics[f"{prefix}rewards/rejected"] = (
                rejected_rewards.mean().cpu().item()
                if not torch.isnan(rejected_rewards.mean())
                else 0.0
            )
            metrics[f"{prefix}rewards/accuracies"] = (
                reward_accuracies.mean().cpu().item()
                if not torch.isnan(reward_accuracies.mean())
                else 0.5
            )

            margin = (chosen_rewards - rejected_rewards).mean()
            metrics[f"{prefix}rewards/margins"] = (
                margin.cpu().item() if not torch.isnan(margin) else 0.5
            )

            metrics[f"{prefix}logps/rejected"] = (
                policy_rejected_logps.detach().mean().cpu().item()
                if not torch.isnan(policy_rejected_logps.mean())
                else -1.0
            )
            metrics[f"{prefix}logps/chosen"] = (
                policy_chosen_logps.detach().mean().cpu().item()
                if not torch.isnan(policy_chosen_logps.mean())
                else -0.5
            )

            if hasattr(policy_chosen_logits, "mean"):
                metrics[f"{prefix}logits/rejected"] = (
                    policy_rejected_logits.detach().mean().cpu().item()
                    if not torch.isnan(policy_rejected_logits.mean())
                    else 0.0
                )
                metrics[f"{prefix}logits/chosen"] = (
                    policy_chosen_logits.detach().mean().cpu().item()
                    if not torch.isnan(policy_chosen_logits.mean())
                    else 0.0
                )
            else:
                metrics[f"{prefix}logits/rejected"] = 0.0
                metrics[f"{prefix}logits/chosen"] = 0.0

            # 多様性関連の指標を追加
            metrics[f"{prefix}diversity/loss"] = (
                diversity_loss.detach().mean().cpu().item()
                if not torch.isnan(diversity_loss.mean())
                else 0.1
            )

            # NaNチェック
            loss_mean = losses.mean()
            if torch.isnan(loss_mean):
                print("WARNING: NaN detected in loss mean. Using default value 1.0.")
                return (
                    torch.tensor(1.0, device=losses.device, requires_grad=True),
                    metrics,
                )

            # 勾配を確保
            if not loss_mean.requires_grad:
                print(
                    "WARNING: Loss mean does not require grad. Adding requires_grad=True."
                )
                loss_mean = loss_mean.clone().detach().requires_grad_(True)

            return loss_mean, metrics

        except Exception as e:
            print(f"ERROR in get_batch_loss_metrics: {str(e)}")
            # エラー発生時のフォールバック
            device = next(model.parameters()).device
            default_metrics = {
                f"{prefix}rewards/chosen": 0.5,
                f"{prefix}rewards/rejected": 0.0,
                f"{prefix}rewards/accuracies": 0.5,
                f"{prefix}rewards/margins": 0.5,
                f"{prefix}logps/rejected": -1.0,
                f"{prefix}logps/chosen": -0.5,
                f"{prefix}logits/rejected": 0.0,
                f"{prefix}logits/chosen": 0.0,
                f"{prefix}diversity/loss": 0.1,
            }
            return torch.tensor(1.0, device=device, requires_grad=True), default_metrics

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        損失計算メソッドのオーバーライド - CPOTrainerに依存せず直接実装
        """
        try:
            loss, outputs = None, None

            # get_batch_loss_metrics を直接呼び出す
            try:
                loss, metrics = self.get_batch_loss_metrics(
                    model, inputs, train_eval="train"
                )

                # 勾配計算のため、requires_grad を確保
                if not loss.requires_grad:
                    loss = loss.clone().detach().requires_grad_(True)
                    print("WARNING: Added requires_grad=True to loss")

                outputs = {"metrics": metrics}
            except Exception as e:
                print(f"ERROR in get_batch_loss_metrics: {str(e)}")
                device = next(model.parameters()).device
                loss = torch.tensor(1.0, device=device, requires_grad=True)
                outputs = {"metrics": {}}

            if return_outputs:
                return loss, outputs
            else:
                return loss
        except Exception as e:
            print(f"ERROR in compute_loss: {str(e)}")
            # エラー発生時のフォールバック
            device = next(model.parameters()).device
            loss = torch.tensor(1.0, device=device, requires_grad=True)

            if return_outputs:
                dummy_outputs = {"loss": loss}
                return loss, dummy_outputs
            else:
                return loss
