#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OpenAI評価機能を持つ文章生成機能を追加した多様性SimPOトレーニングのメインスクリプト
"""

import json
import os
import sys

from datasets import Dataset

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
from transformers.integrations import HfDeepSpeedConfig
from trl import KTOConfig


# 多様性パラメータと生成パラメータを持つCPOConfig拡張
@dataclass
class KTOGenerationEvaluationConfig(KTOConfig):
    """
    KTOにOpenAI評価・生成・多様性パラメータを追加した拡張設定
    """

    diversity_weight: Optional[float] = field(
        default=0.05,
        metadata={"help": "多様性損失の全体的な重み付け（0.01〜0.1）"},
    )

    diversity_alpha: Optional[float] = field(
        default=0.1,
        metadata={"help": "多様性損失内のエントロピー項の重み（0〜1）"},
    )

    enable_generation: Optional[bool] = field(
        default=True,
        metadata={"help": "文章生成機能のオン/オフ"},
    )

    generation_interval: Optional[int] = field(
        default=100,
        metadata={"help": "何ステップごとに生成するか"},
    )

    generation_batch_size: Optional[int] = field(
        default=2,
        metadata={"help": "生成時のバッチサイズ"},
    )

    openai_evaluation: Optional[bool] = field(
        default=True,
        metadata={"help": "OpenAI APIによる生成評価のオン/オフ"},
    )

    openai_model: Optional[str] = field(
        default="gpt-4o-2024-08-06",
        metadata={"help": "OpenAI評価モデル名"},
    )


# 拡張したトレーナークラスをインポート
from train_kto_diverse_openai import KTOGenerationEvaluationTrainer

# OpenAI API有効性確認
if "OPENAI_API_KEY" not in os.environ:
    print("WARNING: OPENAI_API_KEY環境変数が設定されていません。")
    print("OpenAI APIを使用するには、以下のコマンドを実行してください：")
    print("export OPENAI_API_KEY='your-api-key'")

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

# LOCAL_RANK取得
local_rank = int(os.environ.get("LOCAL_RANK", 0))
# # 分散学習の初期化
# if not dist.is_initialized():
#     deepspeed.init_distributed()

# # CUDAデバイス設定
# torch.cuda.set_device(local_rank)

global_rank = dist.get_rank()
world_size = dist.get_world_size()


print(
    f"Process info: local_rank={local_rank}, global_rank={global_rank}, world_size={world_size}"
)

# Load configuration
config_path = "configs/config_kto_generation_evaluation2.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Load DeepSpeed config
with open("configs/ds_config_kto2.json", "r") as f:
    ds_config = yaml.safe_load(f)

# Initialize HfDeepSpeedConfig
dschf = HfDeepSpeedConfig(ds_config)


# Create output directory
os.makedirs("output", exist_ok=True)

# Load dataset
print("Loading dataset...")


# データセットの読み込み
# with open(config["dataset"]["dataset_file"], "r", encoding="utf-8") as f:
#     data = json.load(f)
# dataset = Dataset.from_dict(data)
# train_dataset = dataset

# 学習用にフィルタした JSONL を読み込む
train_dataset = load_dataset(
    "json",
    data_files={"train": config["dataset"]["filtered_train_file"]},
    split="train",
)

print("Loading original test_prefs split…")
dataset = load_dataset(config["dataset"]["name"])
test_dataset = dataset["test_prefs"]


def convert_to_kto_format(dataset):
    """KTOトレーナー用のシンプルな形式に変換"""
    processed_data = []

    for example in dataset:
        # chosenの処理
        chosen_content = ""
        if isinstance(example["chosen"], list):
            for msg in example["chosen"]:
                if msg["role"] == "assistant":
                    chosen_content = msg["content"]
                    break
        else:
            chosen_content = str(example["chosen"])

        # rejectedの処理
        rejected_content = ""
        if isinstance(example["rejected"], list):
            for msg in example["rejected"]:
                if msg["role"] == "assistant":
                    rejected_content = msg["content"]
                    break
        else:
            rejected_content = str(example["rejected"])

        # KTO形式のデータを作成
        processed_data.append(
            {
                "prompt": example["prompt"],
                "chosen": chosen_content,
                "rejected": rejected_content,
            }
        )

    return Dataset.from_list(processed_data)


# 必要のないカラムを消去する
train_dataset = train_dataset.remove_columns(["prompt_id"])

# 前処理を追加
print("Preprocessing dataset for KTO format...")
train_dataset = convert_to_kto_format(train_dataset)

# Print a sample
print("Sample data:")
print(train_dataset[0])


# # 勾配チェックポイントの設定に関する変更
# gradient_checkpointing_kwargs = {"use_reentrant": False}

# Load model and tokenizer
print(f"Loading model: {config['model']['name']}...")
tokenizer = AutoTokenizer.from_pretrained(
    config["model"]["name"], use_fast=False, trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    config["model"]["name"],
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model.gradient_checkpointing_enable()  # 勾配チェックポイントを有効化
model.config.use_cache = False  # キャッシュを無効化（メモリ使用量削減）

# 明示的にパラメータの勾配を有効化
for param in model.parameters():
    param.requires_grad = True


# リファレンスモデルのセットアップ
ref_model = AutoModelForCausalLM.from_pretrained(config["model"]["name"])
ref_model.config.pad_token_id = tokenizer.pad_token_id
ref_model.config.use_cache = False
ref_model.gradient_checkpointing_enable()
ref_model.config.use_cache = False

# Initialize wandb
from datetime import datetime

# タイムスタンプ付きの run_name を作成
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"GenerationEvaluation_{timestamp}"

wandb.init(project="elyza-llama-kto", name=run_name)
wandb.config.update(config)
wandb.watch(model, log="all", log_freq=10)


# Setup training arguments - 拡張した設定クラスを使用
training_args = KTOGenerationEvaluationConfig(
    output_dir="./output/kto-generation-eval",
    loss_type="kto",  # 明示的にkto指定（もしくはデフォルトのままでもOK）
    # KTOの追加パラメータ
    beta=config["training"].get("beta", 0.1),
    desirable_weight=config["training"].get("desirable_weight", 1.0),
    undesirable_weight=config["training"].get("undesirable_weight", 0.1),
    # 多様性や生成、OpenAI評価用パラメータ
    diversity_weight=config["training"]["diversity_weight"],
    diversity_alpha=config["training"]["diversity_alpha"],
    enable_generation=True,
    generation_interval=1,
    generation_batch_size=2,
    openai_evaluation=True,
    openai_model="gpt-4o-2024-08-06",
    # 共通パラメータ
    per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
    num_train_epochs=config["training"]["num_train_epochs"],
    logging_steps=config["training"]["logging_steps"],
    deepspeed="configs/ds_config_kto2.json",
    gradient_checkpointing=config["training"]["gradient_checkpointing"],
    gradient_checkpointing_kwargs={"use_reentrant": False},
    save_strategy=config["training"]["save_strategy"],
    save_steps=config["training"]["save_steps"],
    evaluation_strategy=config["training"]["evaluation_strategy"],
    eval_steps=config["training"]["eval_steps"],
    learning_rate=float(config["training"]["learning_rate"]),
    report_to=config["training"]["report_to"],
    gradient_accumulation_steps=4,
    bf16=True,
    ddp_find_unused_parameters=False,  # DDPの未使用パラメータチェックを無効化
    no_cuda=False,  # CUDAの使用を有効化
    run_name=run_name,  # wandbのrun_name
)

# Create trainer - DiversitySimPOTrainer2WithGenerationを使用
print("Setting up trainer with generation and OpenAI evaluation capability...")

trainer = KTOGenerationEvaluationTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    # train_dataset=formatted_train_dataset,
    # eval_dataset=formatted_test_dataset,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)

# Train model
print("Starting training with periodic text generation and OpenAI evaluation...")
trainer.train()

# Save model
print("Saving model...")
trainer.save_model("./output/generation-evaluation-trainer")
tokenizer.save_pretrained("./output/generation-evaluation-trainer")

print("Training and evaluation completed successfully!")
