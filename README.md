Instruction to activate GPU

condor_submit -interactive -append 'request_gpus=1' -append 'request_cpus=1' -append 'request_memory=16GB'

apptainer exec --nv /.automount/net_rw/net__data_ttk/soshaw/apptainer_images/gabbro.sif bash
source /opt/conda/bin/activate

export LOG_DIR=/.automount/net_rw/net__data_ttk/soshaw/gabbro_logs
export COMET_API_TOKEN=JQ7QMBsBHDB7qau3iihEW0Ion
export HYDRA_FULL_ERROR=1

PYTHONPATH=$(pwd) python gabbro/train.py experiment=example_experiment_tokenization_transformer

Seeds: 42, 2, 17, 73, 109, 256