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
    diversity_weight: float = field(
        default=0.05, metadata={"help": "Weight for the diversity (entropy) loss."}
    )
    diversity_alpha: float = field(
        default=1.0,
        metadata={"help": "Temperature for entropy calculation in diversity loss."},
    )
    sub_batch_size: int = field(
        default=0,
        metadata={
            "help": "Size of sub-batches for memory optimization. 0 means no sub-batching."
        },
    )


class SimPOTrainer(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        training_args = kwargs["args"]
        self.gamma = training_args.simpo_gamma
        self.diversity_weight = getattr(training_args, "diversity_weight", 0.05)
        self.diversity_alpha = getattr(training_args, "diversity_alpha", 1.0)
        self.sub_batch_size = getattr(training_args, "sub_batch_size", 0)
        self.memory_efficient = self.sub_batch_size > 0

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
            # 語彙サイズが大きいため、メモリ効率の良い方法でエントロピーを計算
            entropy_sum = 0.0
            seq_len = policy_chosen_logits.size(1)
            vocab_size = policy_chosen_logits.size(2)

            # シーケンスごとに処理してメモリ使用量を削減
            for i in range(seq_len):
                # 現在のトークン位置のロジットを取得
                pos_logits = policy_chosen_logits[:, i, :]

                # 正規化温度でソフトマックスを計算
                pos_probs = F.softmax(pos_logits / self.diversity_alpha, dim=-1)
                pos_log_probs = F.log_softmax(pos_logits / self.diversity_alpha, dim=-1)

                # エントロピーを計算して合計に追加
                pos_entropy = -torch.sum(pos_probs * pos_log_probs, dim=-1)
                entropy_sum += pos_entropy

                # 不要なテンソルを削除
                del pos_logits, pos_probs, pos_log_probs, pos_entropy

            # 平均エントロピーを計算
            entropy = entropy_sum / seq_len
            diversity_loss = -torch.mean(entropy)
            losses = losses + self.diversity_weight * diversity_loss

        chosen_rewards = self.beta * policy_chosen_logps.detach()
        rejected_rewards = self.beta * policy_rejected_logps.detach()

        return losses, chosen_rewards, rejected_rewards

    def process_sub_batch(self, model, sub_batch, device):
        """サブバッチを個別に処理して結果を返す"""
        sub_concatenated_batch = self.concatenated_inputs(
            sub_batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=device,
        )

        len_chosen = sub_batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "labels": sub_concatenated_batch["concatenated_labels"],
                "decoder_input_ids": sub_concatenated_batch.pop(
                    "concatenated_decoder_input_ids", None
                ),
            }
            if self.is_encoder_decoder
            else {}
        )

        # torch.autocast を使用して混合精度を活用（メモリ使用量削減）
        dtype = (
            torch.float16
            if hasattr(model, "dtype") and model.dtype == torch.float16
            else torch.float32
        )
        with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
            outputs = model(
                sub_concatenated_batch["concatenated_input_ids"],
                attention_mask=sub_concatenated_batch["concatenated_attention_mask"],
                use_cache=False,
                **model_kwargs,
            )

            all_logits = outputs.logits

            all_logps = self.get_batch_logps(
                all_logits,
                sub_concatenated_batch["concatenated_labels"],
                average_log_prob=True,
                is_encoder_decoder=self.is_encoder_decoder,
                label_pad_token_id=self.label_pad_token_id,
            )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]
        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        # キャッシュクリア
        del sub_concatenated_batch, outputs, all_logits, all_logps
        torch.cuda.empty_cache()

        return chosen_logps, rejected_logps, chosen_logits, rejected_logits

    def concatenated_forward(self, model, batch):
        device = self.accelerator.device

        if not self.memory_efficient:
            # オリジナルの実装（サブバッチ処理なし）
            concatenated_batch = self.concatenated_inputs(
                batch,
                is_encoder_decoder=self.is_encoder_decoder,
                label_pad_token_id=self.label_pad_token_id,
                padding_value=self.padding_value,
                device=device,
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

            # torch.autocastを使用してメモリ使用量を削減
            dtype = (
                torch.float16
                if hasattr(model, "dtype") and model.dtype == torch.float16
                else torch.float32
            )
            with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
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

            # 明示的なメモリクリーンアップ
            del concatenated_batch, all_logps
            torch.cuda.empty_cache()

            return chosen_logps, rejected_logps, chosen_logits, rejected_logits
        else:
            # メモリ効率の良いサブバッチ処理
            batch_size = len(batch["chosen_labels"])
            sub_batch_size = min(self.sub_batch_size, batch_size)

            all_chosen_logps = []
            all_rejected_logps = []
            all_chosen_logits = []
            all_rejected_logits = []

            # バッチを小さなサブバッチに分割して処理
            for i in range(0, batch_size, sub_batch_size):
                # サブバッチの作成
                sub_batch = {k: v[i : i + sub_batch_size] for k, v in batch.items()}

                # サブバッチの処理
                chosen_logps, rejected_logps, chosen_logits, rejected_logits = (
                    self.process_sub_batch(model, sub_batch, device)
                )

                # 結果を保存
                all_chosen_logps.append(chosen_logps)
                all_rejected_logps.append(rejected_logps)
                all_chosen_logits.append(chosen_logits)
                all_rejected_logits.append(rejected_logits)

                # 明示的なメモリクリーンアップ
                del (
                    chosen_logps,
                    rejected_logps,
                    chosen_logits,
                    rejected_logits,
                    sub_batch,
                )
                gc.collect()
                torch.cuda.empty_cache()

            # 全サブバッチの結果を結合
            chosen_logps = torch.cat(all_chosen_logps, dim=0)
            rejected_logps = torch.cat(all_rejected_logps, dim=0)
            chosen_logits = torch.cat(all_chosen_logits, dim=0)
            rejected_logits = torch.cat(all_rejected_logits, dim=0)

            # 明示的なメモリクリーンアップ
            del (
                all_chosen_logps,
                all_rejected_logps,
                all_chosen_logits,
                all_rejected_logits,
            )
            gc.collect()
            torch.cuda.empty_cache()

            return chosen_logps, rejected_logps, chosen_logits, rejected_logits

    def get_batch_loss_metrics(self, model, batch, train_eval="train"):
        # デバッグ情報
        if (
            hasattr(self, "state")
            and getattr(self.state, "epoch", None) is not None
            and self.state.global_step % 10 == 0
        ):
            current_mem = torch.cuda.memory_allocated() / (1024**3)
            max_mem = torch.cuda.max_memory_allocated() / (1024**3)
            print(
                f"Memory usage before forward: current={current_mem:.2f}GB, max={max_mem:.2f}GB"
            )

        # 明示的なメモリクリーンアップ
        gc.collect()
        torch.cuda.empty_cache()

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)

        # デバッグ情報
        if (
            hasattr(self, "state")
            and getattr(self.state, "epoch", None) is not None
            and self.state.global_step % 10 == 0
        ):
            current_mem = torch.cuda.memory_allocated() / (1024**3)
            max_mem = torch.cuda.max_memory_allocated() / (1024**3)
            print(
                f"Memory usage after forward: current={current_mem:.2f}GB, max={max_mem:.2f}GB"
            )

        losses, chosen_rewards, rejected_rewards = self.simpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits=policy_chosen_logits,
        )

        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        prefix = "eval_" if train_eval == "eval" else ""

        # CPUにデータを移動して計算（GPUメモリを節約）
        metrics = {
            f"{prefix}rewards/chosen": chosen_rewards.detach().cpu().mean().item(),
            f"{prefix}rewards/rejected": rejected_rewards.detach().cpu().mean().item(),
            f"{prefix}rewards/accuracies": reward_accuracies.detach()
            .cpu()
            .mean()
            .item(),
            f"{prefix}rewards/margins": (chosen_rewards - rejected_rewards)
            .detach()
            .cpu()
            .mean()
            .item(),
            f"{prefix}logps/chosen": policy_chosen_logps.detach().cpu().mean().item(),
            f"{prefix}logps/rejected": policy_rejected_logps.detach()
            .cpu()
            .mean()
            .item(),
        }

        # 余分なロギングは必要な場合のみ行う
        if self.state.global_step % 50 == 0:
            metrics.update(
                {
                    f"{prefix}logits/chosen": policy_chosen_logits.detach()
                    .cpu()
                    .mean()
                    .item(),
                    f"{prefix}logits/rejected": policy_rejected_logits.detach()
                    .cpu()
                    .mean()
                    .item(),
                }
            )

        # 明示的なメモリクリーンアップ
        del (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        )
        del chosen_rewards, rejected_rewards, reward_accuracies
        gc.collect()
        torch.cuda.empty_cache()

        return losses.mean(), metrics
