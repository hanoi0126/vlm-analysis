#!/bin/sh
#PJM -L rscgrp=share-debug
#PJM -L gpu=1
#PJM -L elapse=00:30:00
#PJM -g gb20
#PJM -j
#PJM -o logs/comparison/%j.out

module load gcc/8.3.1
module load python/3.10.13
module load cuda/12.2
module unload gcc
module load gcc/12.2.0

source venv/bin/activate
source jobs/import-env.sh .env

cd /work/gb20/b20070/vlm-analysis

# create log directory
LOG_DIR="logs/comparison/$(date '+%Y-%m-%d/%H-%M-%S')"
mkdir -p ${LOG_DIR}

export TRITON_CACHE_DIR=$PWD/.triton_cache
export MPLCONFIGDIR=$PWD/.matplotlib
mkdir -p $HF_HOME $HF_DATASETS_CACHE $TORCH_HOME $TRITON_CACHE_DIR $MPLCONFIGDIR

export CC=$(which gcc)
export CXX=$(which g++)

export PYTHONPATH=$(pwd):$PYTHONPATH
export TOKENIZERS_PARALLELISM=false

echo "=================================================="
echo "開始時刻: $(TZ=Asia/Tokyo date)"
echo "画像あり/なし比較実験を開始します"
echo "=================================================="

python src/scripts/run_comparison.py "$@" >> ${LOG_DIR}/console_output.log 2>&1

echo "=================================================="
echo "終了時刻: $(TZ=Asia/Tokyo date)"
echo "比較実験が完了しました！"
echo "=================================================="

mv "logs/comparison/${PJM_JOBID}.out" "${LOG_DIR}/job.log"