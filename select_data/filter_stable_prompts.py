import numpy as np
import pandas as pd
from datasets import load_dataset

# パラメータ
VARIANCE_THRESHOLD = 2.0  # 分散の閾値を 2.0 に設定
TARGET_DIM = "helpfulness"
STABLE_CSV = "./select_data/stable_high_variance_prompts.csv"

# プレーン版読み込み
print("Loading UltraFeedback plain dataset…")
ds = load_dataset("openbmb/UltraFeedback", split="train")

# 各サンプルごとに helpfulness スコアを集めて分散を計算
print("Computing per-prompt variances…")
records = []
for idx, example in enumerate(ds):
    instr = example["instruction"]
    ratings = []
    for comp in example.get("completions", []):
        ann = comp.get("annotations", {})
        entry = ann.get(TARGET_DIM)
        if isinstance(entry, dict):
            r = entry.get("Rating")
        elif isinstance(entry, list) and entry:
            r = entry[0].get("Rating")
        else:
            r = None
        if r is not None:
            try:
                ratings.append(float(r))
            except ValueError:
                pass

    # スコアが2件以上あるものだけ
    if len(ratings) >= 2:
        var = np.var(ratings)
        # ③ ここで閾値フィルタ
        if var >= VARIANCE_THRESHOLD:
            records.append(
                {
                    "instruction": instr,
                    "variance": var,
                    "count": len(ratings),
                    "index": idx,
                }
            )

print(f"Selected {len(records)} prompts with variance ≥ {VARIANCE_THRESHOLD}")

# ④ DataFrame にまとめて保存
df = pd.DataFrame(records)
df.to_csv(STABLE_CSV, index=False)
print(f"High-variance prompts saved to {STABLE_CSV}")

# 先頭5件を表示
print(df.head())
