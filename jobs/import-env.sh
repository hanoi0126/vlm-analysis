#!/bin/bash
for kv in $(< $1)
do
  # 空行やコメント行をスキップ
  if [[ -z "$kv" ]] || [[ "$kv" =~ ^[[:space:]]*$ ]] || [[ "$kv" =~ ^# ]]; then
    continue
  fi
  export $kv
done