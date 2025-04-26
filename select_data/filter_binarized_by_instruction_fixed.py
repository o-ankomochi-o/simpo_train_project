# filter_binarized_by_instruction_fixed.py

import json

import pandas as pd
from datasets import load_dataset

# パラメータ
STABLE_CSV = "./select_data/stable_high_variance_prompts.csv"
OUTPUT_JSONL = "./select_data/filtered_train_prefs.jsonl"


def main():
    # 安定プロンプトの読み込み
    stable_df = pd.read_csv(STABLE_CSV)
    stable_instructions = set(stable_df["instruction"].tolist())
    print(f"Loaded {len(stable_instructions)} stable instructions")

    # rain_prefs 読み込み
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    print(f"Loaded train_prefs with {len(ds)} samples")

    # フィルタリング
    def is_stable(ex):
        return ex["prompt"] in stable_instructions

    filtered = ds.filter(is_stable)
    print(f"Filtered down to {len(filtered)} stable prefs")

    # JSONL 形式で保存
    filtered.to_json(OUTPUT_JSONL, lines=True, orient="records")

    print(f"Saved filtered prefs to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
