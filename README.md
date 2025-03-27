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
