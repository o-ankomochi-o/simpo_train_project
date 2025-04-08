#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多様性指標を組み込んだSimPOトレーニングのメインスクリプト
"""

import os
import sys

# Flex-Attention を無効化
os.environ["TRANSFORMERS_NO_FLEX_ATTENTION"] = "1"
# torch._dynamo を無効化
import torch._dynamo

torch._dynamo.disable()

from dataclasses import dataclass, field
from typing import Optional

import deepspeed
import torch
import torch.distributed as dist
import wandb
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import CPOConfig


# 多様性パラメータを持つCPOConfig拡張
@dataclass
class DiversityCPOConfig(CPOConfig):
    """
    多様性指標を追加したCPOConfig拡張
    """

    # 多様性重み - 多様性損失の全体的な重み付け
    diversity_weight: Optional[float] = field(
        default=0.05,
        metadata={"help": "多様性損失の全体的な重み付け。0.01〜0.1の範囲が推奨。"},
    )

    # 多様性アルファ - エントロピー項とKL項のバランス
    diversity_alpha: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "多様性損失内のエントロピー項の重み。0〜1の範囲。"
            "1に近いほどエントロピーが強調され、0に近いほどKL項が強調される。"
        },
    )


# 拡張したトレーナークラスをインポート
# from diversity_simpo_trainer import DiversitySimPOTrainer

from diversity_simpo_trainer2 import DiversitySimPOTrainer2 as DiversitySimPOTrainer

# バージョン情報の表示
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


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
config_path = "configs/config_diversity_simpo.yaml"
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

# Initialize wandb
from datetime import datetime

# タイムスタンプ付きの run_name を作成
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"DiversitySimPO_training_run_{timestamp}"

wandb.init(project="elyza-llama-simpo", name=run_name)
wandb.config.update(config)
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
formatted_train_dataset = formatted_train_dataset.select(range(200))
formatted_test_dataset = formatted_test_dataset.select(range(100))

# Remove unnecessary columns
columns_to_remove = list(
    set(formatted_train_dataset.column_names) - {"prompt", "chosen", "rejected"}
)
formatted_train_dataset = formatted_train_dataset.remove_columns(columns_to_remove)
formatted_test_dataset = formatted_test_dataset.remove_columns(columns_to_remove)

# Setup training arguments - 拡張した設定クラスを使用
training_args = DiversityCPOConfig(
    output_dir="./output/diversity-simpo-trained-model",
    loss_type="simpo",
    cpo_alpha=0.0,  # 純粋なSimPO
    simpo_gamma=config["training"]["simpo_gamma"],
    # 多様性パラメータを追加
    diversity_weight=config["training"]["diversity_weight"],
    diversity_alpha=config["training"]["diversity_alpha"],
    per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
    num_train_epochs=config["training"]["num_train_epochs"],
    logging_steps=config["training"]["logging_steps"],
    deepspeed="configs/ds_config_simpo.json",
    gradient_checkpointing=config["training"]["gradient_checkpointing"],
    save_strategy=config["training"]["save_strategy"],
    save_steps=config["training"]["save_steps"],
    evaluation_strategy=config["training"]["evaluation_strategy"],
    eval_steps=config["training"]["eval_steps"],
    learning_rate=float(config["training"]["learning_rate"]),
    report_to=config["training"]["report_to"],
)

# Create trainer
print("Setting up trainer...")
trainer = DiversitySimPOTrainer(
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
trainer.save_model("./output/diversity-simpo-trained-model")
tokenizer.save_pretrained("./output/diversity-simpo-trained-model")

print("Training completed successfully!")
