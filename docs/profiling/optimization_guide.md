# Transform Profiling: Key Insights & Decisions

**Test Date:** 2026-01-05
**Hardware:** NVIDIA RTX 500 Ada (3.7GB), 16-core CPU, 30.8GB RAM
**Sample:** ISLES_BBS sub-001

---

## Executive Summary

✅ **Profiling completed successfully**
🎯 **Task 3 (Caching) priority:** **HIGH** (would eliminate 53% of transform time)
⚡ **Quick win:** Use 64³ patches instead of 96x128x96 (44% faster)
⚠️ **Bottleneck:** Rand3DElasticd scales poorly with patch size

---

## Key Findings

### 1. Spatial Size Matters Significantly

| Configuration | Spatial Size | Total Time | Speedup vs 96x128x96 |
|---------------|--------------|------------|---------------------|
| unetr_cc_96x128x96 | 96x128x96 | **552ms** | baseline |
| unetr_cc_64x64x64 | 64³ | **311ms** | **1.77x faster** ✅ |
| unetr_cc_patches | Full size | **2143ms** | 0.26x (4x slower) ⚠️ |

**Insight:** Smaller patches = much faster transforms!

### 2. Transform Bottlenecks Identified

**For unetr_cc_96x128x96 (default config):**
1. **LoadImaged**: 255ms (46%) - Disk I/O
2. **Rand3DElasticd**: 179ms (32%) - Expensive augmentation
3. **RandRicianNoised**: 31ms (6%)

**For unetr_cc_64x64x64 (smaller patches):**
1. **LoadImaged**: 239ms (77%) - Dominates!
2. **Rand3DElasticd**: 40ms (13%) - Much faster with smaller patches
3. Others: <10ms each

**For unetr_cc_patches (full size):**
1. **Rand3DElasticd**: 1088ms (51%) - HUGE bottleneck!
2. **LoadImaged**: 204ms (10%)
3. Others: <200ms each

**Insight:** Rand3DElasticd time scales with patch volume (64³ → 96³ → full size)

### 3. Caching Impact Estimate

**Cacheable transforms** (non-random, will only run once):
- LoadImaged
- EnsureChannelFirstd
- ResizeWithPadOrCropd
- MyNormalizeIntensityd (both instances)
- Binarized (both instances)
- CoordConvd

**Non-cacheable transforms** (random, run every epoch):
- Rand3DElasticd
- RandRicianNoised
- RandGibbsNoised
- RandKSpaceSpikeNoised
- RandBiasFieldd
- RandHistogramShiftd
- RandFlipd
- RandShiftIntensityd

**Estimated impact for unetr_cc_96x128x96:**
- Cacheable time: ~294ms (53%)
- Non-cacheable time: ~258ms (47%)
- **First epoch:** 552ms (all transforms)
- **Later epochs:** ~258ms (only random transforms)
- **Expected speedup:** **2.14x** after first epoch ⚡

**Conclusion:** **Caching is HIGH PRIORITY** (Task 3)

---

## Optimization Recommendations

### Immediate Actions (No Code Changes)

**1. Use 64³ patches instead of 96x128x96**
- 44% faster transform time (552ms → 311ms)
- Trade-off: Smaller receptive field (may affect model performance)
- Test impact on validation metrics

**2. Reduce Rand3DElasticd probability**
Current: `prob=0.05` (5% of batches)
- Already low, but consider reducing to 0.02 if time-critical
- Or reduce complexity: `magnitude_range=(1,5)` instead of `(3,10)`

### High Priority (Task 3)

**3. Re-enable caching**
- Expected speedup: 2.14x after first epoch
- Saves 294ms per sample (~53% of transform time)
- **Priority:** HIGH based on profiling results
- **Implementation:** Task 3 (planned)

### Medium Priority (Task 2.1)

**4. Integrate profiling into lesseg_unet**
- Allow users to profile their own data
- Help users choose optimal spatial size
- Guide optimization decisions
- **Priority:** MEDIUM (nice-to-have, not critical)

---

## Task Priority Updates

### Task 2 (Resource Profiling)
**Current focus:** Transform profiling ✅ DONE
**Next:** Full resource profiling (GPU/CPU/memory/disk)
- **Priority:** MEDIUM-LOW (transforms profiled, need full context)
- Can be deferred if Task 3 shows good results

### Task 3 (Re-enable Caching)
**Status:** Blocked by Task 2 profiling results
**Priority:** ~~MEDIUM~~ → **HIGH** (based on profiling)
- Caching would eliminate 53% of transform time
- 2.14x speedup after first epoch
- Should be next priority after documenting profiling results

### Task 2.1 (Integrate Profiling Feature)
**Status:** New subtask
**Priority:** MEDIUM
**Rationale:** Useful feature, but not critical
- Helps users optimize their configs
- Good for documentation/examples
- Can be done after Task 3

---

## Recommended Task Order

```
Current: Task 2.0 (Transform Profiling) ✅ DONE
   ↓
Next: Document profiling results ← YOU ARE HERE
   ↓
Then: Task 3 (Re-enable Caching) ← HIGH PRIORITY
   ↓
After: Task 2.1 (Integrate Profiling) ← Nice-to-have
   ↓
Later: Task 2 (Full Resource Profiling) ← If needed
```

---

## Documentation Needed

### 1. Copy Results to Package Docs
```bash
mkdir -p docs/profiling
cp profiling_results_20260105_184600.md docs/profiling/baseline_results.md
```

### 2. Create User Guide
`docs/profiling_guide.md`:
- How to interpret profiling results
- When to use 64³ vs 96³ patches
- Expected transform times for different configs
- Caching impact explanation

### 3. Update README
- Mention profiling capability (when Task 2.1 implemented)
- Link to profiling guide

---

## Files to Keep/Archive

**Keep in `.lad_work/task2_resource_profiling/`:**
- ✅ `profiling_results_20260105_184600.md` - Successful profiling run
- ✅ `profiling_results_20260105_184600.json` - Raw data
- ✅ `KEY_INSIGHTS.md` - This file
- ✅ `TASK_2.1_INTEGRATION_PLAN.md` - Integration plan
- ✅ `PROFILING_APPROACHES.md` - Design decisions
- ✅ `RUN_TESTS.md` - How to run tests
- ✅ `offline_transform_profiler.py` - Core implementation (for Task 2.1)
- ✅ `run_profiling_tests.py` - Test runner (for Task 2.1)

**Delete (obsolete):**
- ✅ Failed profiling runs (cleaned up)

**Copy to package docs/ (when ready):**
- `profiling_results_20260105_184600.md` → `docs/profiling/baseline_results.md`

---

## Next Steps

1. ✅ Mark Task 2.0 as completed
2. 📝 Document profiling results in package docs/
3. 🔄 Update Task 3 priority to HIGH
4. 🚀 Begin Task 3 (Re-enable Caching) - Expected 2.14x speedup!
5. 📋 Plan Task 2.1 (Integrate Profiling) for later

---

Created: 2026-01-05
Status: Analysis complete, decisions made
Next: Task 3 (Caching) - HIGH PRIORITY
