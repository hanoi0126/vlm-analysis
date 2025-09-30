#!/bin/bash
echo "=================================================="
echo "開始時刻: $(TZ=Asia/Tokyo date)"
echo "=================================================="

export PYTHONPATH=$(pwd):$PYTHONPATH

uv run python src/scripts/run_experiment.py "$@"

echo "=================================================="
echo "終了時刻: $(TZ=Asia/Tokyo date)"
echo "実験が完了しました！"
echo "=================================================="
