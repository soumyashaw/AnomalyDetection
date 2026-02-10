#!/bin/bash
echo "Job started on: $(hostname)"
nvidia-smi

IMG="/.automount/net_rw/net__data_ttk/soshaw/apptainer_images/gabbro.sif"

# Set your wandb API key
export WANDB_API_KEY="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"

# Go to your project folder in home
cd $HOME/AnomalyDetection

apptainer exec --nv $IMG bash -c "
    source /opt/conda/bin/activate
    python train_custom.py --gpu_id 0 --seed 42 --jet_name both --merge_strategy attention --use_wandb
"

echo "Job finished."
