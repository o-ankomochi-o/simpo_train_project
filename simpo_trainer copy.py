# simpo_trainer.py

import gc
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
import wandb
from trl import DPOConfig, DPOTrainer


@dataclass
class SimPOConfig(DPOConfig):
    simpo_gamma: float = field(
        default=0.7, metadata={"help": "Reward margin scaling factor for SimPO."}
    )


class SimPOTrainer(DPOTrainer):
    """
    DPOTrainerをベースにした、シンプルなSimPOトレーナー。
    メモリ使用量を最適化し、必要に応じて拡張できます。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        training_args = kwargs["args"]
        self.gamma = getattr(training_args, "simpo_gamma", 0.7)
        self.use_deepspeed = True

    def compute_loss(self, model, inputs, return_outputs=False):
        # トレーニング前に明示的にメモリをクリーンアップ
        gc.collect()
        torch.cuda.empty_cache()

        # 親クラスのcompute_lossメソッドを呼び出す
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

        # ここでSimPOの損失に修正する
        if hasattr(self, "gamma") and self.gamma != 0.0:
            # SimPOの修正: gamma/betaをlogitsから引く
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

    def log(self, logs, start_time=None):
        # 親クラスのlogメソッドを呼び出し
        super(SimPOTrainer, self).log(logs)

        # wandbにログを記録
        if self.args.report_to == "wandb":
            wandb.log(logs)
