#!/usr/bin/env bash
# Train SNN + CNN for several seeds -> runs/seeds/<seed>/{snn,cnn}.pt + logs
# SNNs run in parallel with 1 thread each (the per-time-step loop does not use more threads);
# CNNs run one after another with the remaining cores (convolutions do use threads).
# Usage: ./run_seeds.sh [seeds...]   (default: 0 1 2 3 4);  PY=path/to/python to override
set -e
PY=${PY:-python}
SEEDS=${@:-0 1 2 3 4}
NS=$(echo $SEEDS | wc -w)
CNN_THREADS=$(( $(getconf _NPROCESSORS_ONLN) - NS )); [ $CNN_THREADS -lt 1 ] && CNN_THREADS=1
for s in $SEEDS; do
  mkdir -p runs/seeds/$s
  $PY train.py --model snn --epochs 25 --seed $s --threads 1 > runs/seeds/$s/snn.log 2>&1 &
done
for s in $SEEDS; do
  $PY train.py --model cnn --epochs 40 --seed $s --threads $CNN_THREADS > runs/seeds/$s/cnn.log 2>&1
done
wait
echo "all seeds done"
