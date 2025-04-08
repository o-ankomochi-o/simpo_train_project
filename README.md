# simpo_train_project

SimPO（Simple Preference Optimization

# SimPO Training for ELYZA Llama-3

シンプルな SimPO（Simple Preference Optimization）を使用して ELYZA Llama-3 モデルをファインチューニングするコード

## プロジェクト構造

```
simpo_train_project/
├── train_simpo_elyza_llama3.py           # 実行用メインスクリプト
├── requirements.txt                      # パッケージリスト
├── README.md                             # プロジェクト説明
├── configs/
│   └── config_cpo.yaml                   # 設定ファイル
└── output/                               # モデル出力
    └── simpo-trained-model/
```

## セットアップと使用方法

1. 必要なパッケージをインストール:

```bash
pip install -r requirements.txt
```

2. 学習の実行:

```bash
python train_simpo_elyza_llama3.py
```

## SimPO について

本実装では以下の設定を使用:

- `loss_type="simpo"`
- `cpo_alpha=0.0`
- `simpo_gamma=0.7`

.
├── README.md
├── diversity_simpo_trainer.py
├── train_diversity_simpo.py
├── configs/
│ ├── config_diversity_simpo.yaml
│ └── ds_config.json
└── output/
└── diversity-simpo-trained-model/

# 多様性促進 SimPO トレーニング

このリポジトリには、多様性を促進する機能を組み込んだ SimPO（Simple Preference Optimization）トレーニングの実装が含まれています。

## 概要

標準の SimPO は、選択された応答と拒否された応答の間の対比学習に基づいています。この拡張バージョンでは、テキスト生成の多様性を促進するための追加の損失項を導入しています。

### 主な機能

- **SimPO 損失**: 標準の SimPO 損失を実装
- **多様性損失**: エントロピーベースの多様性促進と KL ダイバージェンスによる分布差の維持
- **統合ロギング**: 多様性指標を含む詳細なロギングと WandB サポート
- **DeepSpeed 互換**: 効率的な分散トレーニングのための DeepSpeed 統合

## ファイル構成

- `diversity_simpo_trainer.py`: 多様性促進 SimPO トレーナーの実装
- `train_diversity_simpo.py`: トレーニングのメインスクリプト
- `configs/config_diversity_simpo.yaml`: トレーニング設定ファイル
- `configs/ds_config.json`: DeepSpeed 設定ファイル

## 使用方法

### 1. 依存関係のインストール

```bash
pip install torch transformers trl datasets wandb deepspeed pyyaml
```

### 2. 設定ファイルのカスタマイズ

`configs/config_diversity_simpo.yaml` を編集して、使用するモデル、データセット、トレーニングパラメータを設定します。

重要なパラメータ:

- `simpo_gamma`: SimPO のマージン係数（デフォルト: 0.7）
- `diversity_weight`: 多様性損失の全体的な重み（デフォルト: 0.05）
- `diversity_alpha`: 多様性損失内のエントロピー項の重み（デフォルト: 0.1）

### 3. トレーニングの実行

```bash
python train_diversity_simpo.py
```

または、DeepSpeed を使用した分散トレーニングの場合:

```bash
deepspeed --num_gpus=4 train_diversity_simpo.py
```

## 多様性促進メカニズム

多様性損失は以下の 2 つの要素から構成されています:

1. **エントロピー最大化**: 選択された応答の確率分布のエントロピーを最大化することで、より均一なトークン選択を促進
2. **KL ダイバージェンス**: 選択された応答と拒否された応答の間の分布の差異を維持

数式:

```
diversity_loss = -alpha * entropy(chosen_probs) + (1 - alpha) * KL(chosen_logits || rejected_probs)
```

最終的な損失関数:

```
total_loss = simpo_loss + diversity_weight * diversity_loss
```

## パラメータチューニング

多様性の程度は以下のパラメータで調整できます:

- `diversity_weight` (0.01〜0.1): 多様性損失の全体的な重み
- `diversity_alpha` (0〜1): エントロピー項の重み (1 に近いほどエントロピーが強調され、0 に近いほど KL 項が強調されます)

## モニタリング

トレーニング中は以下の指標が追跡されます:

- 標準の SimPO 指標（報酬、マージン、精度など）
- `diversity/loss`: 多様性損失の値

WandB ダッシュボードでこれらの指標を視覚化して、多様性と品質のバランスを評価できます。

## ディレクトリ構造

```
.
├── README.md
├── diversity_simpo_trainer.py
├── train_diversity_simpo.py
├── configs/
│   ├── config_diversity_simpo.yaml
│   └── ds_config.json
└── output/
    └── diversity-simpo-trained-model/
```

## 実装詳細

多様性損失は、選択されたサンプルのエントロピーを高めることで、モデルがより多様なテキストを生成するよう促します。一方、KL ダイバージェンス項は選択と拒否の間の区別を維持し、品質を確保します。

```python
# エントロピー計算
chosen_probs = F.softmax(policy_chosen_logits, dim=-1)
chosen_entropy = -(chosen_probs * torch.log(chosen_probs + 1e-10)).sum(dim=-1).mean()

# KLダイバージェンス計算
rejected_probs = F.softmax(policy_rejected_logits, dim=-1)
kl_div = F.kl_div(
    F.log_softmax(policy_chosen_logits, dim=-1),
    rejected_probs,
    reduction='batchmean'
)

# 多様性損失
diversity_loss = -alpha * chosen_entropy + (1 - alpha) * kl_div
```

## ライセンス

## MIT

CPOTrainer の SimPO 実装に関する発見

SimPO 損失の計算方法:

CPOTrainer では simpo_loss というメソッドは存在せず、cpo_loss メソッド内で loss_type="simpo"の場合に特別な処理が行われます
SimPO の場合は特別なパラメータ simpo_gamma が使われています
基本的な損失計算は-F.logsigmoid(self.beta \* logits)という形で行われます

損失計算フロー:

get_batch_loss_metrics メソッドが全体的な損失計算を担当
concatenated_forward で選択および拒否された応答のロジットと対数確率を取得
cpo_loss で実際の損失計算を行う

修正アプローチ
今回の修正では、元の CPOTrainer の実装をできるだけ尊重しつつ、必要最小限の変更で多様性促進機能を追加しました：

最小限の継承:

CPOTrainer を継承し、必要なメソッドだけをオーバーライド
親クラスの get_batch_loss_metrics メソッドを呼び出してから、多様性損失を追加
これにより、TRL ライブラリが更新されても互換性を保ちやすくなります

多様性損失の追加:

エントロピーベースの多様性促進機能を実装
CPOTrainer の標準の損失計算に影響を与えずに多様性損失を追加
親クラスの concatenated_forward 結果を利用して計算を行う

パラメータの渡し方:

標準の CPOConfig を使用し、追加のパラメータを直接追加
これにより DiversitySimPOTrainer が diversity_weight と diversity_alpha を取得できる
