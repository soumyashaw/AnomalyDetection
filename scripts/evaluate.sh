#!/bin/bash
echo "Job started on: $(hostname)"
nvidia-smi

IMG="/.automount/net_rw/net__data_ttk/soshaw/apptainer_images/gabbro.sif"

# Go to your project folder in home
cd $HOME/AnomalyDetection

apptainer exec --nv $IMG bash -c "
    source /opt/conda/bin/activate
    cd \"$HOME/AnomalyDetection\"
    python -m src.eval.evaluate_CASE --checkpoint aachen_head_expts/run_optim_weak_600_pretrain_class10M_freeze_aachen_20260420_113710/checkpoints/epoch\=17_val_argos\=0.5087.ckpt --model_type aachen
    "

echo "Job finished."
