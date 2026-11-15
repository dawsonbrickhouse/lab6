#!/usr/bin/env bash
set -euo pipefail
while tmux -f /exec-daemon/tmux.portal.conf has-session -t "=part5-500-epochs" 2>/dev/null; do sleep 30; done
B="--hidden 1024 1024 1024 1024 --batch-size 1024 --epochs 30 --no-shuffle --weight-decay 0"
python3 lab6.py --name part6_lr_1e-3 --lr 1e-3 --scheduler none $B
python3 lab6.py --name part6_lr_1e-4 --lr 1e-4 --scheduler none $B
python3 lab6.py --name part6_lr_1e-5 --lr 1e-5 --scheduler none $B
python3 lab6.py --name part6_step_1e-3_s10_g0.1 --lr 1e-3 --scheduler step --step-size 10 --gamma 0.1 $B
python3 lab6.py --name part6_cosine_1e-3 --lr 1e-3 --scheduler cosine $B
python3 lab6.py --name part6_onecycle_2e-3 --lr 2e-3 --scheduler onecycle $B
