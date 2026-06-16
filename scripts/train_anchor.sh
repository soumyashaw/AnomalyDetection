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

    python -m src.train.train_anchor_aachen --batch_size 1024 --learning_rate 2e-4 --naming_identifier anchor_baseline_5_unsup_b1024 --guaranteed_signal_per_batch 5 --injection_probability 1.0 --use_wandb
'
echo "Job finished."
# python -m src.train.train_anchor_aachen --batch_size 64 --learning_rate 1e-4 --naming_identifier anchor_baseline_5_unsup --guaranteed_signal_per_batch 5 --injection_probability 1.0 --use_wandb
# python -m src.train.train_anchor_aachen --batch_size 64 --learning_rate 1e-4 --pretrained_ckpt backbone_weights/pretrained_class_1M_each/backbone.ckpt --load_pretrained --freeze_backbone --naming_identifier anchor_pretrain_freeze_5_unsup --guaranteed_signal_per_batch 5 --injection_probability 1.0 --use_wandb
# /.automount/home/home__home3/institut_thp/soshaw/unsupervised_learning/outputs/eval/anomalous_signal.h5
# /.automount/home/home__home3/institut_thp/soshaw/GenVsMem-on-HEP-Data/results/memorized_events_for_aachen.h5