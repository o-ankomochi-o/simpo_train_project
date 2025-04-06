# simpo_trainer.py

import gc
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from trl import DPOConfig, DPOTrainer


@dataclass
class SimPOConfig(DPOConfig):
    simpo_gamma: float = field(
        default=0.7, metadata={"help": "Reward margin scaling factor for SimPO."}
    )


class SimplifiedSimPOTrainer(DPOTrainer):
    """
    DPOTrainerをベースにした、シンプルなSimPOトレーナー。
    多様性損失を取り除き、メモリ使用量を最適化します。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        training_args = kwargs["args"]
        self.gamma = getattr(training_args, "simpo_gamma", 0.7)
        # DPOTrainerのほとんどの機能をそのまま継承

    def compute_loss(self, model, inputs, return_outputs=False):
        # トレーニング前に明示的にメモリをクリーンアップ
        gc.collect()
        torch.cuda.empty_cache()

        # 親クラスのcompute_lossメソッドを呼び出す
        # これはDPOTrainerの通常の損失計算を行う
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

        # ここでSimPOの損失に修正する
        # gamma変更のみ実装した最もシンプルな形式
        if hasattr(self, "gamma") and self.gamma != 0.0:
            # SimPOの修正: 単にgamma/betaをlogitsから引く
            gamma_term = self.gamma / self.beta

            # 既存のロジットを取得して調整
            if "logits" in outputs:
                # ロジットを修正
                original_logits = outputs["logits"]
                adjusted_logits = original_logits - gamma_term
                outputs["logits"] = adjusted_logits

                # 損失を再計算
                if self.loss_type == "sigmoid":
                    loss = (
                        -F.logsigmoid(self.beta * adjusted_logits)
                        * (1 - self.label_smoothing)
                        - F.logsigmoid(-self.beta * adjusted_logits)
                        * self.label_smoothing
                    ).mean()
                else:  # hinge
                    loss = torch.relu(1 - self.beta * adjusted_logits).mean()

        # メモリクリーンアップ
        gc.collect()
        torch.cuda.empty_cache()

        if return_outputs:
            return loss, outputs
        return loss
