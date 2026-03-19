#!/bin/bash
set -euo pipefail

echo "Job started on: $(hostname)"
nvidia-smi

IMG="/.automount/net_rw/net__data_ttk/soshaw/apptainer_images/gabbro.sif"

# Set your wandb API key
export WANDB_API_KEY="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"

# Go to your project folder in home
cd $HOME/AnomalyDetection

apptainer exec --nv "$IMG" bash -lc '
    set -euo pipefail
    source /opt/conda/bin/activate
    cd "$HOME/AnomalyDetection"

    TRAIN_LOG="$(mktemp /tmp/train_aachen_log.XXXXXX)"
    python -m src.train.train_custom_aachen --gpu_id 0 --seed 42 --jet_name both --merge_strategy concat --naming_identifier weak_5k_aachen --use_wandb 2>&1 | tee "$TRAIN_LOG"

    BEST_CKPT="$(sed -n "s/^Best checkpoint: //p" "$TRAIN_LOG" | tail -n 1)"
    if [[ -z "$BEST_CKPT" ]]; then
        echo "Could not parse best checkpoint path from training log: $TRAIN_LOG"
        exit 1
    fi
    if [[ ! -f "$BEST_CKPT" ]]; then
        echo "Parsed checkpoint does not exist: $BEST_CKPT"
        exit 1
    fi

    echo "Using checkpoint for evaluation: $BEST_CKPT"
    python -m src.eval.evaluate --checkpoint "$BEST_CKPT" --model_type aachen
'

echo "Job finished."
