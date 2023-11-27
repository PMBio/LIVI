#!/bin/sh
DIR='results'
if ! [ -d "$DIR" ]; then
   mkdir $DIR
fi

find logs -name 'sim_*' -printf '%P\n' | xargs -I "{experiment}" python scripts/eval_sim.py --checkpoint_dir="logs/{experiment}/multiruns/*/[0-9]*/checkpoints" --out_file="$DIR/{experiment}.csv" --n_jobs=10