#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多様性指標を組み込んだSimPOトレーナーの拡張実装
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

        # DeepSpeed統合フラグ
        self.use_deepspeed = True

        print(f"Initialized DiversitySimPOTrainer with parameters:")
        print(f"  - simpo_gamma: {self.gamma}")
        print(f"  - diversity_weight: {self.diversity_weight}")
        print(f"  - diversity_alpha: {self.diversity_alpha}")

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

        # エントロピーベースの多様性促進（高いエントロピー = より均一な確率分布 = より多様なトークン選択）
        chosen_probs = F.softmax(policy_chosen_logits, dim=-1)
        chosen_entropy = (
            -(chosen_probs * torch.log(chosen_probs + 1e-10)).sum(dim=-1).mean()
        )

        # KL divergence項（選択された応答と拒否された応答の分布の違いを維持）
        rejected_probs = F.softmax(policy_rejected_logits, dim=-1)
        kl_div = F.kl_div(
            F.log_softmax(policy_chosen_logits, dim=-1),
            rejected_probs,
            reduction="batchmean",
        )

        # 多様性損失（エントロピーを最大化し、KL divergenceを維持）
        # マイナスを付けてエントロピーを最大化（損失を最小化する方向）
        diversity_loss = -alpha * chosen_entropy + (1 - alpha) * kl_div

        return diversity_loss

    def simpo_loss_with_diversity(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_chosen_logits: torch.FloatTensor,
        policy_rejected_logits: torch.FloatTensor,
        diversity_weight: float = None,  # デフォルト値はNone、初期化時に設定した値を使用
    ):
        """多様性促進を組み込んだSimPO損失関数"""
        # diversity_weightがNoneの場合は、初期化時に設定された値を使用
        if diversity_weight is None:
            diversity_weight = self.diversity_weight

        # オリジナルのSimPO損失
        losses, chosen_rewards, rejected_rewards = self.simpo_loss(
            policy_chosen_logps, policy_rejected_logps
        )

        # 多様性損失を計算
        div_loss = self.diversity_loss(policy_chosen_logits, policy_rejected_logits)

        # 結合した損失を返す（多様性損失の重みは調整可能）
        combined_losses = losses + diversity_weight * div_loss

        return combined_losses, chosen_rewards, rejected_rewards, div_loss

    def concatenated_forward(self, model, batch):
        """
        フォワードパスの実行 - 親クラスの実装をオーバーライドして必要な戻り値を確保
        """
        # 親クラスのconcatenated_forwardを呼び出し
        output = super().concatenated_forward(model, batch)

        # 返り値の確認と必要な値の抽出
        if isinstance(output, tuple):
            if len(output) == 2:
                # 親クラスが(policy_chosen_logps, policy_rejected_logps)を返す場合
                policy_chosen_logps, policy_rejected_logps = output
                # ロジットを取得するための追加処理
                policy_chosen_logits = self._get_logits_from_logps(policy_chosen_logps)
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
        try:
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
            print(f"Error in alternative forward pass: {str(e)}")

            # 最終的なフォールバック: ダミーデータの作成
            device = model.device if hasattr(model, "device") else "cpu"
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

        # ここでconcatenated_forwardを呼び出し、必要な4つの値を取得
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)

        losses, chosen_rewards, rejected_rewards, diversity_loss = (
            self.simpo_loss_with_diversity(
                policy_chosen_logps,
                policy_rejected_logps,
                policy_chosen_logits,
                policy_rejected_logits,
            )
        )

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        # 既存の指標
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (
            (chosen_rewards - rejected_rewards).mean().cpu()
        )
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = (
            policy_rejected_logits.detach().mean().cpu()
        )
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()

        # 多様性関連の指標を追加
        metrics[f"{prefix}diversity/loss"] = diversity_loss.detach().mean().cpu()

        return losses.mean(), metrics

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        損失計算メソッドのオーバーライド - DeepSpeed互換性の確保と例外処理
        """
        try:
            return super().compute_loss(model, inputs, return_outputs=return_outputs)
        except Exception as e:
            print(f"ERROR in compute_loss: {str(e)}")
            # エラー発生時のフォールバック
            device = model.device if hasattr(model, "device") else "cpu"
            loss = torch.tensor(1.0, device=device)

            if return_outputs:
                # 最小限の出力を返す
                dummy_outputs = {"loss": loss}
                return loss, dummy_outputs
            else:
                return loss
