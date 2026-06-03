#!/bin/bash

echo "Job started on: $(hostname)"
nvidia-smi

IMG="/.automount/net_rw/net__data_ttk/soshaw/apptainer_images/gabbro.sif"

# Set your wandb API key
export WANDB_API_KEY="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"

# Go to your project folder in home
cd $HOME/AnomalyDetection

apptainer exec --nv "$IMG" bash -lc '
    source /opt/conda/bin/activate
    cd "$HOME/AnomalyDetection"

    python -m src.train.train_custom_aachen --jet_name both --merge_strategy concat --naming_identifier weak_2k_concat_seed256 --seed 256 --use_wandb --use_hpc
'

echo "Job finished."
