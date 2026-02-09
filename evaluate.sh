#!/bin/bash
echo "Job started on: $(hostname)"
nvidia-smi

IMG="/.automount/net_rw/net__data_ttk/soshaw/apptainer_images/gabbro.sif"

# Go to your project folder in home
cd $HOME/AnomalyDetection

apptainer exec --nv $IMG bash -c "
    source /opt/conda/bin/activate
    python evaluate.py --checkpoint dijet_expts/run_supervised_attn_seed2_20260203_022016/checkpoints/anomaly_detector_epoch\=71_val_loss\=0.0955.ckpt --gpu_id 0
    "   

echo "Job finished."
