#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SimPO Trainer implementation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from trl import CPOTrainer


@dataclass
class SimPOConfig(CPOConfig):
    """
    Configuration class for SimPO training.
    """

    simpo_gamma: Optional[float] = field(
        default=0.5,
        metadata={"help": "The target reward margin term in SimPO loss."},
    )
    loss_type: Optional[str] = field(
        default="sigmoid",
        metadata={"help": "The loss type to use. One of ['sigmoid', 'hinge']"},
    )


class SimPOTrainer(CPOTrainer):
    """
    SimPO (Simple Policy Optimization) trainer implementation.
    Extends CPOTrainer to implement SimPO loss function.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Pass all other arguments using **kwargs
        training_args = kwargs["args"]
        self.gamma = training_args.simpo_gamma
        self.loss_type = training_args.loss_type
        # Because we'll be using SimPO, we need to set alpha to 0
        self.alpha = 0.0

    def simpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the SimPO loss for a batch of policy model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the SimPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        gamma_logratios = self.gamma / self.beta
        pi_logratios = pi_logratios.to(self.accelerator.device)
        logits = pi_logratios - gamma_logratios

        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge']"
            )

        chosen_rewards = (
            self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
        )
        rejected_rewards = (
            self.beta * policy_rejected_logps.to(self.accelerator.device).detach()
        )

        return losses, chosen_rewards, rejected_rewards

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor
    ]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
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

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override the compute_loss method to use SimPO loss.
        """
        batch = self.prepare_batch(inputs)

        # Forward pass
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)

        # Compute SimPO loss
        losses, chosen_rewards, rejected_rewards = self.simpo_loss(
            policy_chosen_logps, policy_rejected_logps
        )

        # Calculate the mean loss
        loss = losses.mean()

        # Log metrics
        self.log_metrics(
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            chosen_rewards,
            rejected_rewards,
            losses,
        )

        if return_outputs:
            outputs = {
                "chosen_logps": policy_chosen_logps,
                "rejected_logps": policy_rejected_logps,
                "chosen_logits": policy_chosen_logits,
                "rejected_logits": policy_rejected_logits,
            }
            return loss, outputs

        return loss

    def log_metrics(
        self,
        policy_chosen_logps,
        policy_rejected_logps,
        policy_chosen_logits,
        policy_rejected_logits,
        chosen_rewards,
        rejected_rewards,
        losses,
    ):
        """
        Log metrics for the current batch.
        """
        if not self.is_local_process_zero() or not hasattr(self, "state"):
            return

        if self.state.global_step % self.args.logging_steps == 0:
            prefix = "eval_" if self.is_in_eval else ""

            # Log rewards and accuracies
            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            metrics = {}
            metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu().item()
            metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu().item()
            metrics[f"{prefix}rewards/accuracies"] = (
                reward_accuracies.mean().cpu().item()
            )
            metrics[f"{prefix}rewards/margins"] = (
                (chosen_rewards - rejected_rewards).mean().cpu().item()
            )
            metrics[f"{prefix}logps/rejected"] = (
                policy_rejected_logps.detach().mean().cpu().item()
            )
            metrics[f"{prefix}logps/chosen"] = (
                policy_chosen_logps.detach().mean().cpu().item()
            )
            metrics[f"{prefix}logits/rejected"] = (
                policy_rejected_logits.detach().mean().cpu().item()
            )
            metrics[f"{prefix}logits/chosen"] = (
                policy_chosen_logits.detach().mean().cpu().item()
            )
            metrics[f"{prefix}loss/total"] = losses.mean().cpu().item()

            self.log(metrics)
