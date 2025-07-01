# キャッチコピー生成モデル訓練スクリプト

本リポジトリでは、OpenAI 評価を損失関数に組み込んだ以下 4 種類の学習スクリプトを提供しています。  
目的に応じて SimPO または KTO を選択し、訴求重視 or 多様性重視の学習が可能です。

---

## モデル一覧

| モデル名           | 特徴                                               | モデル                       | データ                                         | 学習スクリプト                           | トレーナー                                | 設定ファイル                                     |
| ------------------ | -------------------------------------------------- | ---------------------------- | ---------------------------------------------- | ---------------------------------------- | ----------------------------------------- | ------------------------------------------------ |
| **SimPO + 多様性** | 多様性・訴求・関連性・読みやすさ・総合の評価を統合 | `meta-llama/Meta-Llama-3-8B` | `ultrafeedback_binarized` + `simpo_pairs.json` | `train_simpo_diverse_openai.py`          | `trainer_simpo_openai_multi.py`           | `configs/config_diversity_simpo.yaml`            |
| **SimPO + 訴求点** | 訴求点重視。モデルに ELYZA を使用                  | `elyza/Llama-3-ELYZA-JP-8B`  | `ultrafeedback_binarized` + `simpo_pairs.json` | `train_diversity_simpo_with_feedback.py` | `appeal_generation_evaluation_trainer.py` | `configs/config_diversity_simpo_appeal.yaml`     |
| **KTO + 多様性**   | 多項目評価を KTO ベースで学習                      | `meta-llama/Meta-Llama-3-8B` | `ultrafeedback_binarized`                      | `train_kto_diverse_openai.py`            | `trainer_kto_diverse_openai.py`           | `configs/config_kto_generation_evaluation2.yaml` |
| **KTO + 訴求点**   | 訴求点スコアに基づいて KTO で最適化                | `elyza/Llama-3-ELYZA-JP-8B`  | `kto_feedback.json`                            | `train_kto_appeal_openai.py`             | `trainer_kto_appeal_openai.py`            | `configs/config_kto_generation_evaluation.yaml`  |

---

## 実行例

```bash
deepspeed --num_gpus=4 train_simpo_diverse_openai.py
```

```

---

## その他情報

- OpenAI の評価には `gpt-4o-2024-08-06` を使用します。
- DeepSpeed 構成ファイルは `configs/` にあります（例: `ds_config_simpo.json`）。
- ログ出力には `wandb` を使用します。


```
