#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main script for training ELYZA Llama-3 model with SimPO.
"""

import os

import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import CPOConfig, CPOTrainer

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
model = AutoModelForCausalLM.from_pretrained(config["model"]["name"], device_map="auto")
tokenizer.pad_token = tokenizer.eos_token


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
)

# Create trainer
print("Setting up trainer...")
trainer = CPOTrainer(
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
model.save_pretrained("./output/simpo-trained-model")
tokenizer.save_pretrained("./output/simpo-trained-model")

print("Training completed successfully!")
