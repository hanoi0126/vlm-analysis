# VLM Analysis

Vision-Language Model (VLM) の内部表現を層ごとに分析するプロジェクトです。

## 機能

### 1. Image ON/OFF 比較実験

Vision-Language Modelが画像を使用する場合と、テキストのみを使用する場合で、内部表現がどのように異なるかを分析します。

```bash
python src/scripts/run_comparison.py
```

### 2. Cross-condition Probing (NEW!)

一方の条件（Image ON）で訓練したprobeを、もう一方の条件（Image OFF）でテストすることで、表現空間の幾何構造の違いを定量化します。

```bash
python src/scripts/run_cross_condition.py
```

## セットアップ

### 必要な環境

- Python 3.10+
- CUDA対応GPU（推奨）

### インストール

```bash
# 依存パッケージのインストール
pip install -r requirements.txt
```

## 使用方法

### 1. 設定ファイルの編集

`configs/config.yaml` を編集して、実験設定をカスタマイズします。

```yaml
model:
  name: qwen
  model_id: Qwen/Qwen2.5-VL-3B-Instruct
  
dataset:
  hf_dataset: "Hanoi0126/visual-object-attributes"
  
experiment:
  tasks:
    - color
    - shape
    - size
    - angle
    - location
    - count
    - position
    - occlusion
```

### 2. Image ON/OFF 比較実験の実行

```bash
python src/scripts/run_comparison.py
```

この実験では：
1. Image ON（画像あり）で特徴抽出 + Probing
2. Image OFF（テキストのみ）で特徴抽出 + Probing
3. 比較プロットの生成

結果は `results/` ディレクトリに保存されます：

```
results/
├── color_qwen3b_llmtap_imageon/
│   ├── features_l00.npy
│   ├── features_l01.npy
│   ├── ...
│   ├── labels.npy
│   └── metrics.json
├── color_qwen3b_llmtap_imageoff/
│   ├── features_l00.npy
│   ├── ...
│   └── metrics.json
└── ...
```

### 3. Cross-condition Probing の実行

**前提条件**: まず Image ON/OFF 比較実験を実行して、特徴を抽出しておく必要があります。

```bash
python src/scripts/run_cross_condition.py
```

この実験では：
1. 既に抽出された Image ON と Image OFF の特徴を読み込み
2. 各タスク・各レイヤーで Cross-condition Probing を実行
   - Image ON で訓練 → Image ON/OFF でテスト
   - Image OFF で訓練 → Image OFF/ON でテスト
3. Accuracy Gap（同条件精度 - 異条件精度）を計算
4. 可視化（レイヤーごとのGapプロット、条件間マトリックス）

結果は `results/cross_condition/` に保存されます：

```
results/cross_condition/
├── color/
│   ├── cross_condition_results.json
│   └── cross_condition_summary.npz
├── shape/
│   └── ...
└── summary.csv
```

### 4. 結果の可視化

実験スクリプトは自動的に可視化を生成しますが、保存されたデータから後で再生成することもできます。

```python
import numpy as np
from src.visualization import plot_cross_condition_gaps

# Load saved summary
data = np.load("results/cross_condition/color/cross_condition_summary.npz")

# Plot accuracy gaps across layers
plot_cross_condition_gaps(
    layers=data["layers"],
    gap_A_to_B=data["A_gap"],
    gap_B_to_A=data["B_gap"],
    task="color",
    title_suffix="Qwen2.5-VL-3B",
)
```

## プロジェクト構造

```
vlm-analysis/
├── configs/
│   └── config.yaml           # 実験設定
├── src/
│   ├── config/
│   │   └── schema.py         # 設定スキーマ (Pydantic)
│   ├── data/
│   │   ├── dataset.py        # データセット
│   │   └── collate.py        # バッチ処理
│   ├── models/
│   │   ├── base.py           # ベースモデル
│   │   ├── qwen.py           # Qwen実装
│   │   └── registry.py       # モデルレジストリ
│   ├── probing/
│   │   ├── trainer.py        # Probe訓練
│   │   ├── runner.py         # 実験実行
│   │   └── cross_condition.py # Cross-condition probing (NEW!)
│   ├── scripts/
│   │   ├── run_comparison.py      # Image ON/OFF比較
│   │   └── run_cross_condition.py # Cross-condition実験 (NEW!)
│   └── visualization/
│       └── plots.py          # プロット関数
├── results/                  # 実験結果
└── README.md
```


