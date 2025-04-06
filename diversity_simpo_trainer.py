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

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """多様性促進を含むバッチ損失と指標の計算"""
        metrics = {}

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
        損失計算メソッドのオーバーライド - DeepSpeed互換性の確保
        """
        return super().compute_loss(model, inputs, return_outputs=return_outputs)
