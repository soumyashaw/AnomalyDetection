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

    python -m src.train.train_custom_aachen --seed 42 --jet_name both --merge_strategy concat --naming_identifier weak_1k_batch1024_lrrootN_aachen --learning_rate 1.414e-4 --batch_size 1024 --use_hpc --use_wandb
'

echo "Job finished."

# python -m src.train.train_custom_aachen --seed 42 --jet_name both --merge_strategy concat --naming_identifier weak_10k_pretrain_genclass10M_freeze_aachen --pretrained_ckpt backbone_weights/pretrained_gen_class_10M/backbone.ckpt --load_pretrained --freeze_backbone --use_hpc --use_wandb