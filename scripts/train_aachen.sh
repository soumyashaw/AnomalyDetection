#!/bin/bash

echo "Job started on: $(hostname)"
nvidia-smi

IMG="/rwthfs/rz/cluster/work/rww74320/LHCO/apptainer_images/gabbro.sif"
PATH1="/rwthfs/rz/cluster/work/rww74320/LHCO"
PATH2="/rwthfs/rz/cluster/work/rww74320/LHCO"

# Set your wandb API key
export WANDB_API_KEY="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"

# Go to your project folder in home
cd $HOME/AnomalyDetection

apptainer exec --nv --bind "$PATH1":"$PATH2" "$IMG" bash -lc '
    source /opt/conda/bin/activate
    cd "$HOME/AnomalyDetection"

    python -m src.train.train_custom_aachen --jet_name both --merge_strategy concat --naming_identifier optim_weak_300_pretrain_class10M_freeze_aachen_seed256 --seed 256 --pretrained_ckpt backbone_weights/pretrained_class_1M_each/backbone.ckpt --load_pretrained --freeze_backbone --use_hpc --use_wandb
'
echo "Job finished."