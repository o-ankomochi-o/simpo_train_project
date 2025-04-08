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


class DiversitySimPOTrainer(CPOTrainer):
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
        alpha: float = None,
    ) -> torch.FloatTensor:
        if alpha is None:
            alpha = self.diversity_alpha

        try:
            # 入力テンソルのクリッピングと整形
            if policy_chosen_logits.ndim > 2:
                policy_chosen_logits = policy_chosen_logits.reshape(
                    -1, policy_chosen_logits.shape[-1]
                )
                policy_rejected_logits = policy_rejected_logits.reshape(
                    -1, policy_rejected_logits.shape[-1]
                )

            # 無効な値のチェックと置き換え
            chosen_valid = torch.isfinite(policy_chosen_logits).all(dim=-1)
            rejected_valid = torch.isfinite(policy_rejected_logits).all(dim=-1)
            valid_indices = chosen_valid & rejected_valid

            if not valid_indices.any():
                print("WARNING: No valid logits found.")
                return torch.tensor(0.1, device=policy_chosen_logits.device)

            # 有効な値のみ使用
            chosen_logits = policy_chosen_logits[valid_indices]
            rejected_logits = policy_rejected_logits[valid_indices]

            # 大きすぎる値のクリッピング（数値安定性のため）
            chosen_logits = torch.clamp(chosen_logits, max=50.0)
            rejected_logits = torch.clamp(rejected_logits, max=50.0)

            # ソフトマックスとエントロピー計算
            chosen_probs = F.softmax(chosen_logits, dim=-1)
            # 数値的安定性のための定数を追加
            chosen_entropy = (
                -(chosen_probs * torch.log(chosen_probs + 1e-10)).sum(dim=-1).mean()
            )

            # KL divergence計算
            rejected_probs = F.softmax(rejected_logits, dim=-1)
            kl_div = F.kl_div(
                F.log_softmax(chosen_logits, dim=-1),
                rejected_probs,
                reduction="batchmean",
                log_target=False,
            )

            # 損失計算
            diversity_loss = -alpha * chosen_entropy + (1 - alpha) * kl_div

            # 最終チェック
            if torch.isnan(diversity_loss) or torch.isinf(diversity_loss):
                print("WARNING: NaN or Inf in final diversity loss calculation.")
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
