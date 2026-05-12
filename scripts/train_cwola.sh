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

    python -m src.train.train_cwola_aachen --naming_identifier weak_cwola_pretrained_freeze_aachen --pretrained_ckpt backbone_weights/pretrained_class_1M_each/backbone.ckpt --load_pretrained --freeze_backbone --use_hpc --use_wandb
'

echo "Job finished."

# python -m src.train.train_cwola_aachen --naming_identifier weak_cwola_pretrained_update_aachen --pretrained_ckpt backbone_weights/pretrained_class_1M_each/backbone.ckpt --load_pretrained --update_backbone --use_hpc --use_wandb
# python -m src.train.train_cwola_aachen --naming_identifier weak_cwola_pretrained_freeze_aachen --pretrained_ckpt backbone_weights/pretrained_class_1M_each/backbone.ckpt --load_pretrained --freeze_backbone --use_hpc --use_wandb