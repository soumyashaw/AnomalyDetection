checkpoint="val_argos"
signal="100"
path2=$(realpath --relative-to=. aachen_head_expts/run_weak_"${signal}"_*seed2_*/checkpoints/*"${checkpoint}"*)
path17=$(realpath --relative-to=. aachen_head_expts/run_weak_"${signal}"_*seed17_*/checkpoints/*"${checkpoint}"*)
path73=$(realpath --relative-to=. aachen_head_expts/run_weak_"${signal}"_*seed73_*/checkpoints/*"${checkpoint}"*)

python -m src.eval.evaluate --checkpoint "$path2" --model_type aachen
python -m src.eval.evaluate --checkpoint "$path17" --model_type aachen
python -m src.eval.evaluate --checkpoint "$path73" --model_type aachen