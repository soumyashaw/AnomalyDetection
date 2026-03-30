#!/bin/bash
echo "Job started on: $(hostname)"

IMG="/.automount/net_rw/net__data_ttk/soshaw/apptainer_images/gabbro.sif"
checkpoint_path="/.automount/home/home__home3/institut_thp/soshaw/omnijet_alpha_AD/logs/omnijet-multihead/runs/2026-03-22_12-56-02_lx3agpu1_StormproofManageability_gen_1M/checkpoints/best.ckpt"
output_name="pretrained_gen_1M_each"
plot_path="backbone_weights/$output_name/"

# Set your wandb API key
export WANDB_API_KEY="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"

# Go to your project folder in home
cd $HOME/AnomalyDetection

apptainer exec --nv $IMG bash -c "
    source /opt/conda/bin/activate
    cd \"$HOME/AnomalyDetection\"
    python -m src.eval.extract_backbone --checkpoint $checkpoint_path --output_name $output_name --verify --metadata
    python -m src.eval.eval_backbone --checkpoint $plot_path/backbone.ckpt --model_type pretrained --output_dir $plot_path
    "

echo "Job finished."
