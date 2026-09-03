#!/usr/bin/env bash
set -euo pipefail

if (( $# != 21 )); then
    echo "Expected 21 arguments from train_aachen.sub, received $#." >&2
    exit 2
fi

dataset_path=$1
seed=$2
jet_name=$3
merge_strategy=$4
batch_size=$5
max_steps=$6
learning_rate=$7
train_val_split=$8
n_signal=$9
n_supp_background=${10}
n_background=${11}
embedding_dim=${12}
naming_identifier=${13}
log_dir=${14}
pretrained_ckpt=${15}
load_pretrained=${16}
freeze_backbone=${17}
use_class_weights=${18}
wandb_project=${19}
use_wandb=${20}
use_hpc=${21}

echo "Job started on: $(hostname)"
echo "Configuration: ${naming_identifier} (seed=${seed})"
nvidia-smi

image="/.automount/net_rw/net__data_ttk/soshaw/apptainer_images/gabbro.sif"
project_dir="$HOME/AnomalyDetection"

# Also provide immutable environment defaults. This keeps argparse's default
# evaluation independent of whatever .env contains when the job starts.
export DATASET_PATH="$dataset_path"
export SEED="$seed"
export JET_NAME="$jet_name"
export MERGE_STRATEGY="$merge_strategy"
export BATCH_SIZE="$batch_size"
export MAX_STEPS="$max_steps"
export LEARNING_RATE="$learning_rate"
export TRAIN_VAL_SPLIT="$train_val_split"
export N_JETS_TRAIN_CUSTOM_AACHEN="[$n_signal,$n_supp_background,$n_background]"
export EMBEDDING_DIM="$embedding_dim"
export LOG_DIR_AACHEN="$log_dir"

train_args=(
    --dataset_path "$dataset_path"
    --seed "$seed"
    --jet_name "$jet_name"
    --merge_strategy "$merge_strategy"
    --batch_size "$batch_size"
    --max_steps "$max_steps"
    --learning_rate "$learning_rate"
    --train_val_split "$train_val_split"
    --n_jets_train "$n_signal" "$n_supp_background" "$n_background"
    --embedding_dim "$embedding_dim"
    --naming_identifier "$naming_identifier"
    --log_dir "$log_dir"
    --use_class_weights "$use_class_weights"
    --wandb_project "$wandb_project"
)

if [[ "$load_pretrained" == "true" ]]; then
    train_args+=(--pretrained_ckpt "$pretrained_ckpt" --load_pretrained)
fi
if [[ "$freeze_backbone" == "true" ]]; then
    train_args+=(--freeze_backbone)
fi
if [[ "$use_wandb" == "true" ]]; then
    : "${WANDB_API_KEY:?WANDB_API_KEY was not exported when the jobs were submitted}"
    train_args+=(--use_wandb)
fi
if [[ "$use_hpc" == "true" ]]; then
    train_args+=(--use_hpc)
fi

apptainer exec --nv --pwd "$project_dir" "$image" \
    /opt/conda/bin/python -m src.train.train_custom_aachen "${train_args[@]}"

echo "Job finished."
