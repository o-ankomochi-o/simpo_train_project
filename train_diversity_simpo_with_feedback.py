#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OpenAI評価機能を持つ文章生成機能を追加した多様性SimPOトレーニングのメインスクリプト
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


# 多様性パラメータと生成パラメータを持つCPOConfig拡張
@dataclass
class GenerationDiversityCPOConfig(CPOConfig):
    """
    多様性指標と生成機能、OpenAI評価を追加したCPOConfig拡張
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

    # 生成機能のオン・オフ
    enable_generation: Optional[bool] = field(
        default=True,
        metadata={"help": "トレーニング中の文章生成機能を有効にするかどうか"},
    )

    # 生成間隔 - 何ステップごとに生成を行うか
    generation_interval: Optional[int] = field(
        default=100,
        metadata={"help": "トレーニング中に何ステップごとに文章生成を行うか"},
    )

    # 生成に使用するバッチサイズ
    generation_batch_size: Optional[int] = field(
        default=2,
        metadata={
            "help": "文章生成時に使用するバッチサイズ（小さいほど計算効率が良い）"
        },
    )

    # OpenAI評価機能のオン・オフ
    openai_evaluation: Optional[bool] = field(
        default=True,
        metadata={"help": "OpenAI APIを使用して生成テキストを評価するかどうか"},
    )

    # 使用するOpenAIモデル
    openai_model: Optional[str] = field(
        default="gpt-3.5-turbo",
        metadata={"help": "評価に使用するOpenAIモデル名"},
    )


# 拡張したトレーナークラスをインポート
# from diversity_simpo_trainer import DiversitySimPOTrainer2WithGeneration
from generation_evaluation_trainer import GenerationEvaluationTrainer

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
data_files = {"train": config["dataset"]["feedback_data"]}
ds_dict = load_dataset("json", data_files=data_files)
full_ds = ds_dict["train"]  # ここに 258 レコード入っている想定

# -------------------------------
# ② 80/20 で分割（prompt で層化）
# -------------------------------
split_ds = full_ds.train_test_split(
    test_size=0.20,  # 20 % を eval に
    stratify_by_column="prompt",  # prompt ごとに同じ割合
    seed=42,  # 再現性
)

train_dataset = split_ds["train"]
test_dataset = split_ds["test"]

print(train_dataset.shape, test_dataset.shape)
# → (206, 3) (52, 3)  など


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
run_name = f"GenerationEvaluation_feedbackdata_{timestamp}"

wandb.init(project="llama-simpo", name=run_name)
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
training_args = GenerationDiversityCPOConfig(
    output_dir="./output/generation-evaluation-trainer",
    loss_type="simpo",
    cpo_alpha=0.0,  # 純粋なSimPO
    simpo_gamma=config["training"]["simpo_gamma"],
    # 多様性パラメータを追加
    diversity_weight=config["training"]["diversity_weight"],
    diversity_alpha=config["training"]["diversity_alpha"],
    # 生成パラメータを追加
    enable_generation=True,
    generation_interval=1,  # 25ステップごとに生成（より頻繁にサンプルを確認）
    generation_batch_size=1,  # バッチサイズを1に縮小（計算効率のため）
    # OpenAI評価パラメータ
    openai_evaluation=True,
    openai_model="gpt-4o-2024-08-06",
    # 一般的なトレーニングパラメータ
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

# Create trainer - DiversitySimPOTrainer2WithGenerationを使用
print("Setting up trainer with generation and OpenAI evaluation capability...")
trainer = GenerationEvaluationTrainer(
    model=model,
    args=training_args,
    train_dataset=formatted_train_dataset,
    eval_dataset=formatted_test_dataset,
    processing_class=tokenizer,
)

# Train model
print("Starting training with periodic text generation and OpenAI evaluation...")
trainer.train()

# Save model
print("Saving model...")
trainer.save_model("./output/generation-evaluation-trainer")
tokenizer.save_pretrained("./output/generation-evaluation-trainer")

# 生成機能をテスト
print("\n===== トレーニング完了後の生成サンプル と OpenAI評価 =====")
sample_prompts = [
    "人工知能の未来について教えてください。",
    "宇宙旅行の可能性についてどう思いますか？",
    "効果的な学習方法を教えてください。",
]

for i, prompt in enumerate(sample_prompts):
    print(f"\nPrompt {i}: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=256,
            do_sample=True,
            temperature=0.7,
            num_return_sequences=1,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_only = response[len(prompt) :]
    print(f"Response: {response_only}")
    print("-" * 60)

print("Training and evaluation completed successfully!")
