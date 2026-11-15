#!/usr/bin/env bash
set -euo pipefail
B="--hidden 1024 1024 1024 1024 --epochs 5 --lr 1e-3 --scheduler none --no-shuffle --weight-decay 0"
for bs in 32 128 512 4096 16384; do
  python3 lab6.py --name "part4_bs_${bs}" --batch-size $bs $B 2>&1 | grep -E "RESULT|Error"
done
