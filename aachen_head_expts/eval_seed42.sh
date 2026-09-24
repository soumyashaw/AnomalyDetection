checkpoint="argos"
seed="17"
path1k=$(realpath --relative-to=. aachen_head_expts/run_optim_weak_1k_*seed"${seed}"_*/checkpoints/*"${checkpoint}"*)
path2k=$(realpath --relative-to=. aachen_head_expts/run_optim_weak_2k_*seed"${seed}"_*/checkpoints/*"${checkpoint}"*)
path5k=$(realpath --relative-to=. aachen_head_expts/run_optim_weak_5k_*seed"${seed}"_*/checkpoints/*"${checkpoint}"*)
path10k=$(realpath --relative-to=. aachen_head_expts/run_optim_weak_10k_*seed"${seed}"_*/checkpoints/*"${checkpoint}"*)
path100=$(realpath --relative-to=. aachen_head_expts/run_optim_weak_100_*seed"${seed}"_*/checkpoints/*"${checkpoint}"*)
path150=$(realpath --relative-to=. aachen_head_expts/run_optim_weak_150_*seed"${seed}"_*/checkpoints/*"${checkpoint}"*)
path300=$(realpath --relative-to=. aachen_head_expts/run_optim_weak_300_*seed"${seed}"_*/checkpoints/*"${checkpoint}"*)
path500=$(realpath --relative-to=. aachen_head_expts/run_optim_weak_500_*seed"${seed}"_*/checkpoints/*"${checkpoint}"*)
path600=$(realpath --relative-to=. aachen_head_expts/run_optim_weak_600_*seed"${seed}"_*/checkpoints/*"${checkpoint}"*)
path700=$(realpath --relative-to=. aachen_head_expts/run_optim_weak_700_*seed"${seed}"_*/checkpoints/*"${checkpoint}"*)
path900=$(realpath --relative-to=. aachen_head_expts/run_optim_weak_900_*seed"${seed}"_*/checkpoints/*"${checkpoint}"*)


python -m src.eval.evaluate --checkpoint "$path1k" --model_type aachen
python -m src.eval.evaluate --checkpoint "$path2k" --model_type aachen
python -m src.eval.evaluate --checkpoint "$path5k" --model_type aachen
python -m src.eval.evaluate --checkpoint "$path10k" --model_type aachen
python -m src.eval.evaluate --checkpoint "$path100" --model_type aachen
python -m src.eval.evaluate --checkpoint "$path150" --model_type aachen
python -m src.eval.evaluate --checkpoint "$path300" --model_type aachen
python -m src.eval.evaluate --checkpoint "$path500" --model_type aachen
python -m src.eval.evaluate --checkpoint "$path600" --model_type aachen
python -m src.eval.evaluate --checkpoint "$path700" --model_type aachen
python -m src.eval.evaluate --checkpoint "$path900" --model_type aachen