# Catastrophic Forgetting Study: Golden Batch Injection + Latent Space Tracking

## Hypothesis

When training with extremely rare signal jets (e.g., 1% signal fraction in weak supervision), the model experiences **catastrophic forgetting**: 
- A gradient update from a rare signal batch **temporarily forms** signal representations in the latent space
- This is immediately **erased** by the next pure-QCD background batch
- Result: Signal representations never stabilize, leading to poor anomaly detection performance

This experiment **visually and mathematically proves** this erasure by:
1. Training normally for 99 steps
2. Enriching training at step 100 with a "Golden Batch" (50% signal / 50% background)
3. Tracking latent space with UMAP + t-SNE at key moments:
   - Before golden batch (step 99)
   - During golden batch (step 100)
   - After recovery (step 101+)

## Data Integrity

**Strict data separation maintained:**
- **Training data**: `sn_25k_SR_train.h5` + `bg_100k_SR_supp.h5` (used with golden batch injection)
- **Validation data**: Independent train/val split from above
- **Tracking set** (independent): 100 signal + 100 background from TEST files (`sn_50k_SR_test.h5`, `bg_200k_SR_test.h5`)
- **No data leakage** between train/val/test

## Usage

### Basic Command

```bash
python -m src.poc_expts.expt_catastrophic_forgetting_aachen \
  --study_catastrophic_forgetting \
  --signal_test_path /path/to/sn_50k_SR_test.h5 \
  --bg_test_path /path/to/bg_200k_SR_test.h5 \
  --jet_name both \
  --merge_strategy concat \
  --load_pretrained \
  --pretrained_ckpt /path/to/pretrained_backbone.pt \
  --max_steps 200 \
  --learning_rate 1e-4 \
  --batch_size 64 \
  --gpu_id 0 \
  --use_wandb
```

### Full Example with All Arguments

```bash
python -m src.poc_expts.expt_catastrophic_forgetting_aachen \
  --dataset_path /mnt/SAS_B/Soumya/datasets/LHCO \
  --study_catastrophic_forgetting \
  --golden_injection_step 100 \
  --signal_test_path /mnt/SAS_B/Soumya/datasets/LHCO/sn_50k_SR_test.h5 \
  --bg_test_path /mnt/SAS_B/Soumya/datasets/LHCO/bg_200k_SR_test.h5 \
  --tracking_set_size 100 \
  --jet_name both \
  --merge_strategy concat \
  --batch_size 64 \
  --max_steps 200 \
  --learning_rate 1e-4 \
  --train_val_split 0.8 \
  --n_jets_train '[2000, 10000, 20000]' \
  --embedding_dim 128 \
  --load_pretrained \
  --pretrained_ckpt /path/to/omnijetnorm_epoch10_step_50000.pt \
  --use_class_weights true \
  --gpu_id 0 \
  --seed 42 \
  --naming_identifier "catastrophic_forgetting_concat" \
  --log_dir aachen_head_expts \
  --use_wandb \
  --wandb_project anomaly-detection-lhco \
  --wandb_entity your-entity
```

## Key Arguments

### Catastrophic Forgetting Study Args
| Argument | Default | Description |
|----------|---------|-------------|
| `--study_catastrophic_forgetting` | `False` | Enable catastrophic forgetting study |
| `--golden_injection_step` | `100` | Step at which to inject golden batch |
| `--signal_test_path` | `None` | Path to test signal H5 (REQUIRED for study) |
| `--bg_test_path` | `None` | Path to test background H5 (REQUIRED for study) |
| `--tracking_set_size` | `100` | Number of signal and background jets in tracking set |

### Recommended Settings
- **max_steps**: 200 (100 before + 100 after golden injection)
- **tracking_set_size**: 100 (balanced 100 signal + 100 background)
- **golden_injection_step**: 100 (exactly at halfway point)

## Output Structure

```
aachen_head_expts/run_catastrophic_forgetting_concat_YYYYMMDD_HHMMSS/
├── config.json                           # All hyperparameters
├── results.json                          # Final metrics
├── summary.log                           # Training summary
├── checkpoints/                          # Model checkpoints
└── catastrophic_forgetting_tracking/     # Latent space analysis
    ├── latent_space_step00001_umap.png   # UMAP before golden batch
    ├── latent_space_step00001_tsne.png   # t-SNE before golden batch
    ├── latent_space_step00100_umap.png   # UMAP during golden batch
    ├── latent_space_step00100_tsne.png   # t-SNE during golden batch
    ├── latent_space_step00101_umap.png   # UMAP after golden batch
    ├── latent_space_step00101_tsne.png   # t-SNE after golden batch
    ├── latent_space_step00200_umap.png   # UMAP at end
    ├── latent_space_step00200_tsne.png   # t-SNE at end
    └── metrics.json                      # Clustering quality metrics
```

## Tracking Timeline

The model's latent representations are captured and visualized at these steps:

| Step | Event | What to Look For |
|------|-------|-----------------|
| **1** | Training starts | Baseline: mixed/scattered clusters |
| **99** | Before golden batch | Still poor separation (rare signal drowning) |
| **100** | Golden batch injected | **Signal cluster suddenly forms!** |
| **101** | After golden batch | Signal cluster **rapidly dissolves** (catastrophic forgetting) |
| **~100** | Midpoint | Degraded separation again |
| **200** | End of training | Final state (compare to step 1) |

## Metrics Computed

At each tracking step, two separation metrics are recorded:

### Silhouette Score
- **Definition**: Measures how well-separated clusters are
- **Range**: [-1, 1] where 1 = perfect separation, 0 = overlapping, -1 = wrong clusters
- **Interpretation**: Higher is better; drop after step 100 indicates forgetting

### Davies-Bouldin Index
- **Definition**: Average similarity between each cluster and its most similar neighbor
- **Range**: [0, ∞) where 0 = perfect separation
- **Interpretation**: Lower is better; increase after step 100 indicates forgetting

Both metrics are saved to `metrics.json` and show the **temporal collapse** of signal representations.

## Visualization Details

### UMAP (Uniform Manifold Approximation and Projection)
- Fast, interpretable 2D projection
- Preserves both local and global structure
- Better for seeing cluster formation
- Each point = one jet (red=signal, blue=background)

### t-SNE (t-Distributed Stochastic Neighbor Embedding)
- High-quality separations for perceptual interpretation
- Slower but often clearer clusters
- Better for distinguishing within-cluster details
- Same coloring: red=signal, blue=background

## Example Interpretation

### What You Should See

**Step 1-99 (Baseline Weak Supervision)**
```
[Mixed blob]  - Signal and background overlapping
  • Low silhouette: ~0.1-0.3
  • High davies_bouldin: ~1.5-2.0
```

**Step 100 (Golden Batch Injection)**
```
[Signal cluster emerges!]
  • Silhouette jumps to: ~0.4-0.6
  • Davies-Bouldin drops to: ~1.0-1.2
  • Visual: Red and blue clearly separate
```

**Step 101-199 (Recovery/Forgetting)**
```
[Signal cluster collapses]
  • Silhouette drops back to: ~0.1-0.3
  • Davies-Bouldin climbs to: ~1.5+
  • Visual: Clusters re-merge into blob
```

**Step 200 (End of Training)**
```
[Final state]
  • Depends on whether model recovered with more data
  • If catastrophic forgetting: Similar to step 1
  • If partial learning: Slight improvement over step 1
```

## Requirements

### Python Packages
```bash
pip install umap-learn scikit-learn matplotlib torch lightning wandb awkward
```

### File Structure
```
/path/to/dataset/
├── sn_25k_SR_train.h5      # Training signal (golden batch)
├── bg_100k_SR_supp.h5      # Training background (golden batch + normal training)
├── bg_200k_SR_train.h5     # Training background (normal training)
├── sn_50k_SR_test.h5       # TEST signal (MUST for tracking set)
└── bg_200k_SR_test.h5      # TEST background (MUST for tracking set)
```

## Advanced Usage

### Multiple Injection Points
To study forgetting at different timescales:

```bash
# Quick forgetting test (inject at step 50)
python -m src.poc_expts.expt_catastrophic_forgetting_aachen \
  --study_catastrophic_forgetting \
  --golden_injection_step 50 \
  --max_steps 100 \
  ...

# Long-term recovery test (inject at step 500)
python -m src.poc_expts.expt_catastrophic_forgetting_aachen \
  --study_catastrophic_forgetting \
  --golden_injection_step 500 \
  --max_steps 1000 \
  ...
```

### Vary Golden Batch Composition
Edit `load_golden_batch()` to test imbalanced ratios:

```python
# 75% signal / 25% background instead of 50/50
golden_batch = load_golden_batch(
    ...,
    n_signal=48,
    n_background=16,
    ...
)
```

### Tracking Different Merge Strategies
```bash
# Compare concat vs average vs attention
for merge_strat in concat average weighted_sum attention; do
  python -m src.poc_expts.expt_catastrophic_forgetting_aachen \
    --study_catastrophic_forgetting \
    --merge_strategy $merge_strat \
    --naming_identifier "catastrophic_forgetting_$merge_strat" \
    ...
done
```

## Troubleshooting

### "Failed to extract embeddings via hook"
- Check if `BackboneAachenClassificationLightning` has a `backbone` attribute
- Verify model is in eval mode during latent extraction
- Ensure tracking set has correct jet format (dijet if `--jet_name both`)

### "Test data files not found"
- Verify `--signal_test_path` and `--bg_test_path` point to valid H5 files
- These MUST be different from training data to avoid leakage

### UMAP not installed
- Install with: `pip install umap-learn`
- t-SNE will still work as fallback

### Visualizations not saving
- Check `aachen_head_expts/run_*/catastrophic_forgetting_tracking/` directory exists
- Verify write permissions to log directory
- Check disk space

## Expected Runtime

- **Data Loading**: ~2-3 minutes (test sets)
- **200 steps training**: ~15-30 minutes (depends on GPU)
- **Per latent extraction** (6 times): ~1-2 minutes each
- **Total**: ~40-60 minutes with latent tracking enabled

## Scientific Value

This experiment provides:

1. **Visual proof** of catastrophic forgetting (UMAP/t-SNE collapse)
2. **Quantitative metrics** (silhouette, davies_bouldin drop)
3. **Training timeline** showing exact moment of forgetting
4. **Architecture agnostic** (works with any merge strategy)
5. **Reproducible** (fixed test sets, logged random seed)

## Citation & References

If using this study in your thesis or publication:

```
@thesis{
  title={Catastrophic Forgetting in Weakly-Supervised Anomaly Detection},
  author={Your Name},
  year={2026},
  school={Your Institution},
  note={Golden Batch Injection Experiment}
}
```

Related papers:
- McCloskey & Cohen (1989) - Catastrophic Forgetting
- Kirkpatrick et al. (2017) - Elastic Weight Consolidation
- Rusu et al. (2016) - Progressive Neural Networks
