#!/usr/bin/env bash
set -euo pipefail
B="--batch-size 1024 --epochs 5 --lr 1e-3 --scheduler none --no-shuffle --weight-decay 0"
for h in "16" "64" "64 32" "64 128 64" "256 256" "256 512 256" "512 512 512" "1024 1024 1024 1024"; do
  python3 lab6.py --name "part3_arch_${h// /x}" --hidden $h $B 2>&1 | grep -E "RESULT|Error"
done
