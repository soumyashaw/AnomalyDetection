#!/bin/bash
echo "Job started on: $(hostname)"
nvidia-smi

IMG="/.automount/net_rw/net__data_ttk/soshaw/apptainer_images/gabbro.sif"

# Go to your project folder in home
cd $HOME/AnomalyDetection

apptainer exec --nv $IMG bash -c "
    source /opt/conda/bin/activate
    python evaluate_true_roc.py --checkpoint_folder dijet_expts/run_20260312_170307/checkpoints/ --model_type dijet
    "   

echo "Job finished."
