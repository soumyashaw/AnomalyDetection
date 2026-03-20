# Quick Reference: Catastrophic Forgetting Study

## TL;DR

Study whether rare signal jets get **forgotten** after being learned due to subsequent pure-background batches.

## One-Liner Command

```bash
python -m src.poc_expts.expt_catastrophic_forgetting_aachen --study_catastrophic_forgetting --signal_test_path /path/to/sn_50k_SR_test.h5 --bg_test_path /path/to/bg_200k_SR_test.h5 --jet_name both --max_steps 200 --gpu_id 0 --load_pretrained --pretrained_ckpt /path/to/ckpt.pt
```

## Timeline: What Happens

| Step | Batch Type | Expected Metrics | Latent Space |
|------|-----------|-----------------|--------------|
| 1-99 | Normal 1% signal | Silhouette ≈ 0.1-0.3 | Mixed blob |
| **100** | **Golden 50% signal** | **Silhouette ↑ to 0.4-0.6** | **Signal clusters form!** |
| 101-199 | Normal 1% signal | Silhouette ↓ back to 0.1-0.3 | Clusters **collapse** (forgetting) |
| 200 | Normal 1% signal | Final metric | Model state |

## Key Files Generated

```
catastrophic_forgetting_tracking/
├── latent_space_step00001_umap.png     ← Baseline
├── latent_space_step00099_umap.png     ← Before golden batch
├── latent_space_step00100_umap.png     ← Golden batch! (clusters form)
├── latent_space_step00101_umap.png     ← After (clusters collapse)
├── latent_space_step00100_tsne.png     ← Same for t-SNE
├── latent_space_step00101_tsne.png
└── metrics.json                         ← Scores over time
```

## Visual Interpretation

### Good Signs of Catastrophic Forgetting
✅ Step 99: Red & blue mixed/overlapping  
✅ Step 100: Red & blue **suddenly separate**  
✅ Step 101: Red & blue **rapidly re-merge**  
✅ Silhouette: 0.1 → 0.5 → 0.1 (spike then drop)

### Signs of Partial Recovery
⚠️ Step 101+: Silhouette stays elevated (> 0.3)  
⚠️ Suggests model is learning to detect signal better over time

## Required Arguments

```bash
--study_catastrophic_forgetting         # Enable study
--signal_test_path <H5_file>            # Test signal (e.g., sn_50k_SR_test.h5)
--bg_test_path <H5_file>                # Test background (e.g., bg_200k_SR_test.h5)
```

## Optional Arguments

```bash
--golden_injection_step 100             # When to inject (default: 100)
--tracking_set_size 100                 # Tracking set size per class (default: 100)
--max_steps 200                         # Total steps (at least 2× injection point)
```

## Metrics Explained

### Silhouette Score (Higher = Better)
- -1 to +1 range
- Drop after golden batch = **forgetting signal**
- Numbers: 0.1-0.3 (poor), 0.4-0.6 (good), 0.7+ (excellent)

### Davies-Bouldin Index (Lower = Better)
- 0 to ∞ range
- Increase after golden batch = **forgetting**
- Numbers: 1.0-1.5 (good), 1.5-2.0 (moderate), 2.0+ (poor)

## Data Integrity Checklist

- ✅ Golden batch from **TRAINING** data
- ✅ Tracking set from **TEST** data (different from training)
- ✅ No data leakage between sets
- ✅ Validation split independent

## Debug Mode

```bash
# Reduce max_steps for quick test
--max_steps 20 --golden_injection_step 10

# Enable W&B for real-time monitoring
--use_wandb --wandb_project anomaly-detection-lhco

# Smaller tracking set for speed
--tracking_set_size 20
```

## Success Criteria

**Catastrophic Forgetting Proven** if:
1. Silhouette jumps at step 100 (e.g., 0.2 → 0.5)
2. Silhouette drops after step 100 (e.g., 0.5 → 0.2)
3. Davies-Bouldin shows inverse pattern
4. UMAP/t-SNE visualizations show cluster formation then collapse

**Partial Recovery** if:
1. Silhouette still elevated at step 150+ (> 0.3)
2. Model learning to detect signal with more training
3. Could indicate need for gradient accumulation / more signal

## Expected Resource Usage

| Resource | Time | GPU Memory |
|----------|------|-----------|
| Data loading | 3-5 min | ~2 GB |
| Training 200 steps | 15-30 min | ~8-10 GB |
| Latent tracking (6×) | 6-12 min | ~4-5 GB |
| **Total** | **25-50 min** | **~10-12 GB** |

## Common Pitfalls

❌ Using training test files for tracking → Data leakage  
❌ golden_injection_step > max_steps/2 → Insufficient post-injection steps  
❌ tracking_set_size too large → OOM during extraction  
❌ Forgetting test paths → Will crash

## Next Steps After Experiment

If catastrophic forgetting **confirmed**:
→ Run `expt_grad_accumulation_aachen.py` to test gradient accumulation as fix  
→ Run `expt_batching_aachen.py` with guaranteed signal injection

If catastrophic forgetting **not confirmed**:
→ Model already handles rare signal well  
→ Check if gradient accumulation still improves performance  
→ Investigate feature quality with `expt_diagnostic_ablation.py`

## Comparison to Other Approaches

| Approach | Proves Forgetting | Visual | Quantitative |
|----------|------------------|--------|--------------|
| Catastrophic Forgetting Study | ✅ **YES** | UMAP/t-SNE | Silhouette ✓ |
| Diagnostic Ablation (guaranteed signal) | ❌ No | N/A | AUC only |
| Gradient Accumulation | ⚠️ Indirect | N/A | AUC improvement |

---

**Scientific Value**: First direct visual + quantitative proof of catastrophic forgetting in weakly-supervised anomaly detection with 1% signal fraction.
