#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"

# Read only the W&B secret. The list values in this project's .env are dotenv
# syntax, not shell syntax, so sourcing the whole file would be unsafe.
if [[ -z ${WANDB_API_KEY:-} && -f .env ]]; then
    while IFS='=' read -r key value; do
        key=${key#export }
        [[ "$key" == "WANDB_API_KEY" ]] || continue
        value=${value%$'\r'}
        value=${value#\"}
        value=${value%\"}
        value=${value#\'}
        value=${value%\'}
        export WANDB_API_KEY="$value"
        break
    done < .env
fi

: "${WANDB_API_KEY:?Add WANDB_API_KEY to .env or export it before submission}"
mkdir -p batchSubmissions
exec condor_submit scripts/train_aachen.sub
