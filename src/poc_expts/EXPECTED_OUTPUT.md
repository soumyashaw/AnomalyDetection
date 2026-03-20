# Expected Output & Interpretation Guide

## Console Output During Execution

### Startup Phase
```
================================================================================
Loading Golden Batch (50% signal / 50% background)
================================================================================

Loading 32 signal jets from /path/to/sn_25k_SR_train.h5...
Loading 32 background jets from /path/to/bg_100k_SR_supp.h5...
Golden batch created: 32 signal + 32 background = 64 total

================================================================================
Loading Tracking Set from TEST data (independent from training)
================================================================================

Loading 100 signal jets from /path/to/sn_50k_SR_test.h5...
Loading 100 background jets from /path/to/bg_200k_SR_test.h5...
Tracking set created: 100 signal + 100 background = 200 total
⚠️  Data from TEST set (completely independent)

================================================================================
Catastrophic Forgetting Study ENABLED
  Golden batch injection at step: 100
  Tracking set size: 200 jets (signal + background)
================================================================================
```

### Training Output

#### Steps 1-99 (Normal Weak Supervision Phase)
```
Global step 1: loss=0.6923, val_argos=0.0234, val_auc=0.5123
Global step 2: loss=0.6891, val_argos=0.0241, val_auc=0.5189
...
Global step 99: loss=0.4521, val_argos=0.0567, val_auc=0.5876
```

#### Step 100 (GOLDEN BATCH INJECTION)
```
================================================================================
🟡 INJECTING GOLDEN BATCH at Step 100
================================================================================

Global step 100: loss=0.3124, val_argos=0.1234, val_auc=0.6234
     ↑ Loss drops significantly (enriched batch)
     ↑ ARGOS improves (more signal exposure)

================================================================================
LATENT SPACE TRACKING at Step 100
================================================================================

Computing UMAP...
Computing t-SNE...
Signal-Background Separation Metrics:
  Silhouette Score: 0.4823 (↑ better)
  Davies-Bouldin Index: 1.0456 (↓ better)
================================================================================
```

#### Steps 101-199 (Recovery/Forgetting Phase)
```
Global step 101: loss=0.6123, val_argos=0.0589, val_auc=0.5923
     ↑ Loss spikes back up (normal 1% batch)
     ↑ ARGOS drops (signal representations erasing)

Global step 102: loss=0.5834, val_argos=0.0521, val_auc=0.5645

================================================================================
LATENT SPACE TRACKING at Step 101
================================================================================

Computing UMAP...
Computing t-SNE...
Signal-Background Separation Metrics:
  Silhouette Score: 0.1876 (↓ worse - CLUSTER COLLAPSE!)
  Davies-Bouldin Index: 1.8234 (↑ worse - CLUSTERS RE-MERGING)
================================================================================

Global step 150: loss=0.5456, val_argos=0.0512, val_auc=0.5678
```

#### Step 200 (Final State)
```
================================================================================
LATENT SPACE TRACKING at Step 200
================================================================================

Computing UMAP...
Computing t-SNE...
Signal-Background Separation Metrics:
  Silhouette Score: 0.1734
  Davies-Bouldin Index: 1.9012

Latent space tracking complete! Saved to <run_dir>/catastrophic_forgetting_tracking/metrics.json
================================================================================
```

## Output Files

### 1. metrics.json
```json
{
    "silhouette": {
        "1": 0.1532,
        "99": 0.1687,
        "100": 0.4823,
        "101": 0.1876,
        "100": 0.1823,
        "200": 0.1734
    },
    "davies_bouldin": {
        "1": 1.8234,
        "99": 1.7956,
        "100": 1.0456,
        "101": 1.8234,
        "100": 1.8312,
        "200": 1.9012
    }
}
```

### 2. UMAP Visualizations

**latent_space_step00001_umap.png** (Baseline)
```
[Visual: Scattered blue and red dots, heavily overlapping]
Legend: Red=Signal, Blue=Background
Status: Mixed blob, poor separation
Silhouette: 0.15
```

**latent_space_step00099_umap.png** (Before Golden Batch)
```
[Visual: Still scattered blue and red dots, no clusters]
Legend: Red=Signal, Blue=Background
Status: Signal drowning in noise, no clear clusters
Silhouette: 0.17
```

**latent_space_step00100_umap.png** (DURING GOLDEN BATCH ← KEY!)
```
[Visual: RED CLUSTER ON LEFT, BLUE CLUSTER ON RIGHT]
✨ SUDDEN SEPARATION! ✨
Legend: Red=Signal (forming cluster), Blue=Background (distinct cluster)
Status: Signal representation EMERGED
Silhouette: 0.48 ↑↑↑
```

**latent_space_step00101_umap.png** (AFTER GOLDEN BATCH ← KEY!)
```
[Visual: Blue and red dots re-merging into central blob]
↓ COLLAPSE ↓
Legend: Red=Signal (dissolving), Blue=Background (re-mixing)
Status: Signal cluster RAPIDLY ERASED (catastrophic forgetting!)
Silhouette: 0.19 ↓↓↓
```

**latent_space_step00100_umap.png** (Midpoint)
```
[Visual: Back to scattered mixed blob]
Legend: Red=Signal, Blue=Background
Status: No recovery, still poor separation
Silhouette: 0.18
```

**latent_space_step00200_umap.png** (Final State)
```
[Visual: Scattered mixed blob, similar to step 1]
Legend: Red=Signal, Blue=Background
Status: Model learned little from gold injection alone
Silhouette: 0.17
```

### 3. t-SNE Visualizations

**latent_space_step00100_tsne.png**
```
[Visual: Two tight clusters separated]
- Red cluster (upper left)
- Blue cluster (lower right)
- Clear boundary between them
Description: t-SNE reveals fine-grained separation from UMAP
Silhouette: Same as UMAP (~0.48)
```

**latent_space_step00101_tsne.png**
```
[Visual: Clusters deteriorating, merging]
- Red and blue increasingly overlapping
- Clear boundary blurring
- Transition to mixed state
Description: t-SNE shows forgetting progression
Silhouette: Same as UMAP (~0.19)
```

## Numerical Interpretation

### Silhouette Score Progression

```
PERFECT CATASTROPHIC FORGETTING SIGNATURE:

Step  1-99:   0.15-0.18 (baseline poor)
Step  100:    0.48      (↑3x improvement!)
Step  101-199:0.18-0.20 (↓back to baseline)

INTERPRETATION:
- Step 100 spike: Signal manifold suddenly visible
- Step 101+ drop: Signal manifold erased by background-only batches
- Final plateau: Never recovers (only 200 steps total)
```

### Davies-Bouldin Index Progression

```
INVERSE PATTERN (lower = better):

Step  1-99:   1.80-1.95  (poor separation)
Step  100:    1.04       (↓ clusters very separate)
Step  101-199:1.80-1.95  (↑ back to poor)

INTERPRETATION:
- Step 100 drop: Cluster similarity minimized (clear separation)
- Step 101+ rise: Clusters become similar again (re-merging)
- Same forgetting signal as Silhouette
```

## What Success Looks Like

### Criterion 1: Metric Spike at Step 100
```
PASS ✅ if:
  Silhouette: 0.3-0.6 at step 100 (at least 2× baseline)
  Davies-Bouldin: 0.8-1.5 at step 100 (drop from baseline ~1.8)
```

### Criterion 2: Metric Drop at Step 101
```
PASS ✅ if:
  Silhouette: Returns to 0.1-0.3 after step 100
  Davies-Bouldin: Returns to 1.7-2.0 after step 100
```

### Criterion 3: Visual Cluster Dynamics
```
PASS ✅ if:
  Step 100 UMAP/t-SNE: Clear red and blue clusters
  Step 101 UMAP/t-SNE: Clusters noticeably more mixed
  Process visible: Transition from separated → mixed
```

### Criterion 4: Metrics Don't Recover
```
PASS ✅ if:
  Step 200 metrics ≈ Step 1 metrics
  No improvement after golden injection
  Indicates lasting forgetting effect
```

## Partial Recovery (Unexpected)

```
If silhouette > 0.3 at step 150-200:
  ⚠️ Model is LEARNING TO DETECT SIGNAL
     - Not pure catastrophic forgetting
     - May indicate:
       * Better architecture than expected
       * Feature quality sufficient for recovery
       * Need gradient accumulation AFTER recovery too
     
  Next step: Run gradient accumulation study
```

## Complete Failure (Debug Needed)

```
If silhouette ≈ 0 throughout:
  ❌ Check:
     1. Tracking set loaded correctly
     2. Model output shape compatible with hook
     3. Embeddings extracted (check prints)
     4. H5 files readable and non-empty

If silhouette doesn't spike at step 100:
  ❌ Check:
     1. Golden batch actually injected (see console output)
     2. Golden batch has correct label structure
     3. Random seed not causing identical batches
     4. Learning rate sufficient for weight updates
```

## Visualization Quality Tips

### High Quality Output
```
✅ Random seed set (reproducible)
✅ 200 resolution DPI
✅ Clear color contrast (red vs blue)
✅ Legend visible
✅ Title with step number and method
```

### For Publication

```
Use these for thesis/paper:
1. Silhouette score plot over time (step vs score)
2. Davies-Bouldin plot over time
3. Step 100 UMAP (colored clusters forming)
4. Step 101 UMAP (clusters collapsing)
5. Side-by-side comparison: step 100 vs 101

Caption example:
"Latent space representations at step 100 (golden batch injection)
show sudden signal-background separation (silhouette 0.48), 
which collapses at step 101 (silhouette 0.19), demonstrating 
catastrophic forgetting of rare signal jets in weak supervision."
```

## Reading the Metrics JSON

```json
{
    "silhouette": {
        "1": 0.1532,      ← baseline (poor)
        "99": 0.1687,     ← before injection (still poor)
        "100": 0.4823,    ← SPIKE! (golden batch works)
        "101": 0.1876,    ← DROP! (forgetting occurs)
        "100": 0.1823,    ← middle (no recovery)
        "200": 0.1734     ← end (back to baseline)
    }
}
```

**Summary metric**: (0.4823 - 0.1687) / 0.1687 = **186% spike** = Strong forgetting signal

---

**Expected Runtime**: 40-60 minutes  
**Expected Storage**: 50-100 MB (visualizations)  
**Expected GPU Memory**: 10-12 GB  
**Success Rate**: ~85% (if setup correct)
