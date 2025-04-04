# simpo_trainer.py

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
    diversity_weight: float = field(
        default=0.05, metadata={"help": "Weight for the diversity (entropy) loss."}
    )
    diversity_alpha: float = field(
        default=1.0,
        metadata={"help": "Temperature for entropy calculation in diversity loss."},
    )


class SimPOTrainer(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        training_args = kwargs["args"]
        self.gamma = training_args.simpo_gamma
        self.diversity_weight = getattr(training_args, "diversity_weight", 0.05)
        self.diversity_alpha = getattr(training_args, "diversity_alpha", 1.0)

    def simpo_loss(
        self, policy_chosen_logps, policy_rejected_logps, policy_chosen_logits=None
    ):
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        gamma_logratios = self.gamma / self.beta
        logits = pi_logratios - gamma_logratios

        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        diversity_loss = 0.0
        if policy_chosen_logits is not None:
            probs = F.softmax(policy_chosen_logits / self.diversity_alpha, dim=-1)
            log_probs = F.log_softmax(
                policy_chosen_logits / self.diversity_alpha, dim=-1
            )
            entropy = -torch.sum(probs * log_probs, dim=-1)
            diversity_loss = -torch.mean(entropy)
            losses = losses + self.diversity_weight * diversity_loss

        chosen_rewards = self.beta * policy_chosen_logps.detach()
        rejected_rewards = self.beta * policy_rejected_logps.detach()

        return losses, chosen_rewards, rejected_rewards

    def concatenated_forward(self, model, batch):
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop(
                    "concatenated_decoder_input_ids", None
                ),
            }
            if self.is_encoder_decoder
            else {}
        )

        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        ).logits

        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=True,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]
        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    def get_batch_loss_metrics(self, model, batch, train_eval="train"):
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)

        losses, chosen_rewards, rejected_rewards = self.simpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits=policy_chosen_logits,
        )

        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        prefix = "eval_" if train_eval == "eval" else ""

        metrics = {
            f"{prefix}rewards/chosen": chosen_rewards.mean().cpu(),
            f"{prefix}rewards/rejected": rejected_rewards.mean().cpu(),
            f"{prefix}rewards/accuracies": reward_accuracies.mean().cpu(),
            f"{prefix}rewards/margins": (chosen_rewards - rejected_rewards)
            .mean()
            .cpu(),
            f"{prefix}logps/chosen": policy_chosen_logps.mean().cpu(),
            f"{prefix}logps/rejected": policy_rejected_logps.mean().cpu(),
            f"{prefix}logits/chosen": policy_chosen_logits.mean().cpu(),
            f"{prefix}logits/rejected": policy_rejected_logits.mean().cpu(),
        }

        return losses.mean(), metrics
