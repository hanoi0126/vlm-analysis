#!/bin/bash
echo "=================================================="
echo "開始時刻: $(TZ=Asia/Tokyo date)"
echo "=================================================="

export PYTHONPATH=$(pwd):$PYTHONPATH

uv run python src/scripts/extract_features.py "$@"

echo "=================================================="
echo "終了時刻: $(TZ=Asia/Tokyo date)"
echo "特徴抽出が完了しました！"
echo "=================================================="
