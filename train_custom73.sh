#!/bin/bash
echo "Job started on: $(hostname)"
nvidia-smi

IMG="/.automount/net_rw/net__data_ttk/soshaw/apptainer_images/gabbro.sif"

# Go to your project folder in home
cd $HOME/AnomalyDetection

apptainer exec --nv $IMG bash -c "
    source /opt/conda/bin/activate
    python train.py --gpu_id 0 --seed 73 --jet_name both --merge_strategy attention
"

echo "Job finished."
