#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main script for training ELYZA Llama-3 model with SimPO.
"""

import os
import sys

# Flex-Attention を無効化
os.environ["TRANSFORMERS_NO_FLEX_ATTENTION"] = "1"
# torch._dynamo を無効化
import torch._dynamo

torch._dynamo.disable()

import deepspeed
import torch
import torch.distributed as dist
import wandb
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import CPOConfig, CPOTrainer

# バージョン情報の表示
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


class CustomCPOTrainer(CPOTrainer):
    """DeepSpeedとログ機能を統合したカスタムCPOトレーナー-"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_deepspeed = True

    def compute_loss(self, model, inputs, return_outputs=False):
        return super().compute_loss(model, inputs, return_outputs=return_outputs)

    # def compute_loss(
    #     self, model, inputs, return_outputs=False, num_items_in_batch=None
    # ):
    #     print("🧠 compute_loss called")
    #     # num_items_in_batch を親クラスにそのまま渡す（使わないなら無視される）
    #     return super().compute_loss(
    #         model,
    #         inputs,
    #         return_outputs=return_outputs,
    #         num_items_in_batch=num_items_in_batch,
    #     )

    def log(self, logs, start_time=None):
        # 親クラスのlogメソッドをインスタンスメソッドとして呼び出し
        super(CustomCPOTrainer, self).log(logs)

        # 親クラスの処理後にwandbにログを記録
        if self.args.report_to == "wandb":
            wandb.log(logs)


# 明示的な分散環境の初期化
deepspeed.init_distributed()


# LOCAL_RANKとglobal_rankを設定
local_rank = int(os.environ.get("LOCAL_RANK", 0))
global_rank = dist.get_rank()
world_size = dist.get_world_size()

print(
    f"Process info: local_rank={local_rank}, global_rank={global_rank}, world_size={world_size}"
)
# Load configuration
config_path = "configs/config_cpo.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Create output directory
os.makedirs("output", exist_ok=True)

# Load dataset
print("Loading dataset...")
dataset = load_dataset(config["dataset"]["name"])
train_dataset = dataset["train_prefs"]
test_dataset = dataset["test_prefs"]

# Print a sample
print("Sample data:")
print(train_dataset[0])

# Load model and tokenizer
print(f"Loading model: {config['model']['name']}...")
tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
model = AutoModelForCausalLM.from_pretrained(config["model"]["name"])
tokenizer.pad_token = tokenizer.eos_token

wandb.init(project="test", name="SimPO_training_run")

wandb.watch(model, log="all", log_freq=10)


# Preprocess dataset
def preprocess_function(example):
    # Get prompt
    prompt_text = example["prompt"]

    # Extract assistant responses
    chosen_reply = next(
        (msg["content"] for msg in example["chosen"] if msg["role"] == "assistant"),
        None,
    )
    rejected_reply = next(
        (msg["content"] for msg in example["rejected"] if msg["role"] == "assistant"),
        None,
    )

    # Skip if assistant responses not found
    if chosen_reply is None or rejected_reply is None:
        return {}

    # Return formatted data
    return {
        "prompt": prompt_text,
        "chosen": chosen_reply,
        "rejected": rejected_reply,
    }


print("Preprocessing dataset...")
formatted_train_dataset = train_dataset.map(preprocess_function, batched=False)
formatted_test_dataset = test_dataset.map(preprocess_function, batched=False)

# Remove unnecessary columns
columns_to_remove = list(
    set(formatted_train_dataset.column_names) - {"prompt", "chosen", "rejected"}
)
formatted_train_dataset = formatted_train_dataset.remove_columns(columns_to_remove)
formatted_test_dataset = formatted_test_dataset.remove_columns(columns_to_remove)

# Setup training arguments
training_args = CPOConfig(
    output_dir="./output/simpo-trained-model",
    loss_type="simpo",
    cpo_alpha=0.0,  # For pure SimPO
    simpo_gamma=0.7,
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_steps=10,
    deepspeed="configs/ds_config.json",
    gradient_checkpointing=True,  # Save memory by using gradient checkpointing
    save_strategy="no",
    # save_steps=100,
    evaluation_strategy="steps",
    eval_steps=100,
    learning_rate=5e-6,
    report_to="wandb",
)

# Create trainer
print("Setting up trainer...")
trainer = CustomCPOTrainer(
    model=model,
    args=training_args,
    train_dataset=formatted_train_dataset,
    eval_dataset=formatted_test_dataset,
    processing_class=tokenizer,
)

# Train model
print("Starting training...")
trainer.train()

# Save model
print("Saving model...")
trainer.save_model("./output/simpo-trained-model")
tokenizer.save_pretrained("./output/simpo-trained-model")

print("Training completed successfully!")
