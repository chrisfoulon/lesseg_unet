# Transform Profiling Results

**Test Date:** 2026-01-05 18:46:00
**Sample:** `sub-001_dwi.nii.gz`

## Hardware Configuration

- **Hostname:** ubuntu-laptop
- **CPU:** x86_64
- **CPU Cores:** 16 physical, 22 logical
- **RAM:** 30.8 GB
- **GPU:** NVIDIA RTX 500 Ada Generation Laptop GPU
- **GPU Memory:** 3.7 GB
- **CUDA Version:** 12.6

## Configuration Comparison

| Configuration | Spatial Size | Total Time (ms) | Top Bottleneck | % of Total |
|---------------|--------------|-----------------|----------------|------------|
| unetr_cc_96x128x96 | 96x128x96 | 552.1 | LoadImaged | 46.3% |
| unetr_cc_64x64x64 | 64³ | 311.1 | LoadImaged | 76.9% |
| unetr_cc_patches | None | 2142.7 | Rand3DElasticd | 50.8% |

## Detailed Results

### unetr_cc_96x128x96

**Description:** Standard UNETR-CC with 96x128x96 patches (default)

| Transform | Mean (ms) | Std (ms) | % Total | Min (ms) | Max (ms) |
|-----------|-----------|----------|---------|----------|----------|
| LoadImaged | 255.8 | 9.9 | 46.3% | 243.7 | 281.3 |
| Rand3DElasticd | 179.1 | 36.9 | 32.4% | 135.8 | 271.8 |
| RandRicianNoised | 30.8 | 0.9 | 5.6% | 29.7 | 32.5 |
| MyNormalizeIntensityd | 19.7 | 1.9 | 3.6% | 17.2 | 23.3 |
| RandBiasFieldd | 14.1 | 0.5 | 2.6% | 13.4 | 15.1 |
| RandKSpaceSpikeNoised | 13.6 | 0.8 | 2.5% | 12.5 | 15.0 |
| MyNormalizeIntensityd | 12.7 | 2.6 | 2.3% | 6.5 | 15.2 |
| RandGibbsNoised | 10.9 | 3.5 | 2.0% | 8.6 | 21.3 |
| CoordConvd | 6.7 | 2.7 | 1.2% | 4.1 | 14.7 |
| RandHistogramShiftd | 4.2 | 0.8 | 0.8% | 3.5 | 6.6 |

**Total Transform Time:** 552.1ms

### unetr_cc_64x64x64

**Description:** UNETR-CC with smaller 64x64x64 patches

| Transform | Mean (ms) | Std (ms) | % Total | Min (ms) | Max (ms) |
|-----------|-----------|----------|---------|----------|----------|
| LoadImaged | 239.1 | 13.3 | 76.9% | 219.6 | 254.3 |
| Rand3DElasticd | 40.4 | 12.3 | 13.0% | 28.4 | 66.6 |
| RandRicianNoised | 7.1 | 0.7 | 2.3% | 6.6 | 8.6 |
| MyNormalizeIntensityd | 5.8 | 0.7 | 1.9% | 4.8 | 7.0 |
| RandKSpaceSpikeNoised | 4.6 | 1.4 | 1.5% | 3.7 | 8.8 |
| RandBiasFieldd | 3.0 | 0.2 | 1.0% | 2.7 | 3.4 |
| RandGibbsNoised | 2.9 | 2.0 | 0.9% | 1.8 | 8.7 |
| MyNormalizeIntensityd | 2.4 | 0.4 | 0.8% | 2.0 | 3.0 |
| RandHistogramShiftd | 2.1 | 0.5 | 0.7% | 1.6 | 2.9 |
| ResizeWithPadOrCropd | 1.1 | 0.0 | 0.4% | 1.0 | 1.1 |

**Total Transform Time:** 311.1ms

### unetr_cc_patches

**Description:** UNETR-CC with patches configuration

| Transform | Mean (ms) | Std (ms) | % Total | Min (ms) | Max (ms) |
|-----------|-----------|----------|---------|----------|----------|
| Rand3DElasticd | 1088.2 | 101.5 | 50.8% | 977.9 | 1282.8 |
| LoadImaged | 204.2 | 22.5 | 9.5% | 183.7 | 254.5 |
| RandRicianNoised | 196.5 | 7.2 | 9.2% | 183.3 | 209.7 |
| RandKSpaceSpikeNoised | 154.8 | 4.3 | 7.2% | 143.9 | 160.0 |
| RandGibbsNoised | 115.6 | 7.8 | 5.4% | 104.5 | 128.5 |
| RandBiasFieldd | 103.2 | 9.9 | 4.8% | 95.1 | 131.5 |
| MyNormalizeIntensityd | 91.3 | 3.5 | 4.3% | 82.6 | 96.1 |
| MyNormalizeIntensityd | 88.5 | 3.5 | 4.1% | 81.2 | 94.5 |
| CoordConvd | 51.9 | 21.7 | 2.4% | 40.4 | 116.6 |
| RandHistogramShiftd | 23.2 | 3.7 | 1.1% | 19.2 | 32.8 |

**Total Transform Time:** 2142.7ms

## Optimization Recommendations

2. **💾 Caching would eliminate ~53% of transform time**
   - Task 3 (Re-enable caching) is **HIGH PRIORITY**
