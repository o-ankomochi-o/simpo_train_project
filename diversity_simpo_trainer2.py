#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多様性指標を組み込んだSimPOトレーナーの拡張実装 - 最小限の変更版
"""

from typing import Dict, List, Literal, Union

import torch
import torch.nn.functional as F
import wandb
from trl import CPOTrainer


class DiversitySimPOTrainer2(CPOTrainer):
    """
    多様性促進機能を追加したSimPOトレーナー実装

    CPOTrainerを拡張し、テキスト生成の多様性を促進する機能を追加します。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        training_args = kwargs["args"]

        # 多様性関連のハイパーパラメータ
        self.diversity_weight = getattr(training_args, "diversity_weight", 0.05)
        self.diversity_alpha = getattr(training_args, "diversity_alpha", 0.1)

        print(f"Initialized DiversitySimPOTrainer with parameters:")
        print(f"  - simpo_gamma: {self.simpo_gamma}")
        print(f"  - diversity_weight: {self.diversity_weight}")
        print(f"  - diversity_alpha: {self.diversity_alpha}")
        print(f"  - loss_type: {self.loss_type}")

    def log(self, logs, start_time=None):
        """
        拡張ログ機能 - wandbサポート付き
        """
        # 親クラスのlogメソッドを呼び出し
        super().log(logs)

        # 親クラスの処理後にwandbにも記録
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
                policy_chosen_logits = policy_chosen_logits.reshape(
                    -1, policy_chosen_logits.shape[-1]
                )
                policy_rejected_logits = policy_rejected_logits.reshape(
                    -1, policy_rejected_logits.shape[-1]
                )

            # エントロピーベースの多様性促進（高いエントロピー = より均一な確率分布 = より多様なトークン選択）
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

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """多様性促進を含むバッチ損失と指標の計算

        親クラスの実装を拡張して、多様性損失を追加します。
        """
        # 親クラスの get_batch_loss_metrics を呼び出す
        # これにより、policy_chosen_logps、policy_rejected_logps、および通常の損失が計算されます
        loss, metrics = super().get_batch_loss_metrics(model, batch, train_eval)

        try:
            # concatenated_forward を呼び出して必要な値を取得
            forward_output = self.concatenated_forward(model, batch)

            # unpack the outputs from concatenated_forward
            policy_chosen_logps = forward_output[0]
            policy_rejected_logps = forward_output[1]
            policy_chosen_logits = forward_output[2]
            policy_rejected_logits = forward_output[3]

            # 多様性損失を計算
            div_loss = self.diversity_loss(policy_chosen_logits, policy_rejected_logits)

            # 元の損失に多様性損失を追加
            loss = loss + self.diversity_weight * div_loss

            # 多様性関連の指標を追加
            prefix = "eval_" if train_eval == "eval" else ""
            metrics[f"{prefix}diversity/loss"] = (
                self.accelerator.gather_for_metrics(div_loss).detach().mean().item()
            )

            return loss, metrics

        except Exception as e:
            print(f"ERROR in extended get_batch_loss_metrics: {str(e)}")
            print(f"Falling back to original implementation")
            # エラーが発生した場合は元の実装結果を返す
            return loss, metrics
