# Transform Dictionary - Literature References

This document provides scientific references for the noise models and augmentation
strategies used in the multi-modal transform dictionaries.

## Noise Models in Diffusion MRI

### Rician / Noncentral-Chi Noise

Magnitude MR images have **Rician noise** (single-coil) or **noncentral-chi noise**
(multi-coil). This is NOT Gaussian in the magnitude domain.

- **Reference**: Cardenas-Blanco et al. (2008). "Noise in magnitude magnetic resonance
  images." Concepts in Magnetic Resonance Part A, 32A(6), 409-416.
  [PDF](https://pfeifer.phas.ubc.ca/refbase/files/Cardenas-Blanco-ConceptsInMagnetic-2008-32a-409.pdf)

- **Implication**: Use `RandRicianNoised` instead of Gaussian noise for DWI augmentation.

### ADC Noise Characteristics

ADC is computed as: `ADC ≈ -1/b × ln(Sb/S0)`

This **nonlinear transformation** makes ADC noise:
- Heteroscedastic (variance depends on signal level)
- Non-Gaussian
- Biased at low SNR due to the magnitude noise floor

- **Reference**: Koay & Basser (2006). "Analytically exact correction scheme for signal
  extraction from noisy magnitude MR signals." Journal of Magnetic Resonance, 179(2), 317-322.
  [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1090780708003698)

- **Implication**: Noise augmentation on ADC should account for heteroscedasticity.
  When possible, augment upstream (DWI level) and recompute.

### Noise in Averaged DWI (TRACE)

TRACE images are computed by averaging across diffusion directions. The resulting
image inherits magnitude-domain non-Gaussian noise with reduced variance.

- **Reference**: Pieciak et al. (2017). "Noise estimation from averaged diffusion
  weighted images: A Rician to Gaussian approach." Magnetic Resonance in Medicine.
  [PMC4282362](https://pmc.ncbi.nlm.nih.gov/articles/PMC4282362/)

---

## Artifact Models

### Gibbs Ringing

Gibbs ringing (truncation artifact) occurs at sharp edges due to k-space truncation.
It affects both DWI and ADC maps, and can cause **negative ADC values** at sharp edges.

- **Reference**: Veraart et al. (2016). "Gibbs ringing in diffusion MRI."
  Magnetic Resonance in Medicine, 76(1), 301-314.
  [PMC4915073](https://pmc.ncbi.nlm.nih.gov/articles/PMC4915073/)
  [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.25866)

- **Implication**: Apply `RandGibbsNoised` to BOTH DWI and ADC (propagates through calculation).

### K-Space Spike (Herringbone) Artifact

Spike artifacts from scanner instability create structured stripe-like patterns.
These propagate from DWI to ADC maps.

- **Reference**: Radiopaedia. "Herringbone artifact."
  [Radiopaedia](https://radiopaedia.org/articles/herringbone-artifact)

- **Implication**: Apply `RandKSpaceSpikeNoised` to both modalities, but with
  reduced probability for ADC (indirect effect).

### Bias Field (B1 Inhomogeneity)

Smooth multiplicative intensity variations from coil sensitivity and B1 issues.
In ADC (ratio/log), bias field **partially cancels** but residuals often remain.

- **Reference**: MRtrix3 Documentation. "dwibiascorrect."
  [MRtrix Docs](https://mrtrix.readthedocs.io/en/dev/reference/commands/dwibiascorrect.html)

- **Implication**: Apply `RandBiasFieldd` to DWI with moderate strength.
  For ADC, use reduced coefficients (residual effects only).

### EPI Susceptibility Distortion

DWI is commonly acquired with EPI, which is sensitive to magnetic field
inhomogeneities causing geometric distortions.

- **Reference**: Irfanoglu et al. (2022). "EPI susceptibility correction introduces
  significant differences in diffusion MRI metrics." Magnetic Resonance Imaging.
  [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0730725X22000819)

- **Note**: This is typically corrected in preprocessing, not augmented.

---

## Augmentation Strategy Summary

### Per-Modality Transforms (BEFORE concatenation)

| Transform | DWI | ADC | Reference |
|-----------|-----|-----|-----------|
| Normalization | Yes | Yes | Required for each modality's range |
| `RandHistogramShiftd` | Yes | Yes | Different histograms |
| `RandGibbsNoised` | Yes | Yes | [Veraart 2016] |
| `RandKSpaceSpikeNoised` | Yes (p=0.1) | Yes (p=0.05) | [Radiopaedia] |

### Shared Transforms (AFTER concatenation)

| Transform | Notes | Reference |
|-----------|-------|-----------|
| `Rand3DElasticd` | Same deformation to all | Spatial consistency |
| `RandRicianNoised` | channel_wise=True | [Cardenas-Blanco 2008] |
| `RandBiasFieldd` | Different pattern/channel | [MRtrix Docs] |

### Modality-Specific Parameters

| Transform | DWI Params | ADC Params | Rationale |
|-----------|------------|------------|-----------|
| `RandRicianNoised` | std=0.03 | std=0.02 | ADC noise is propagated, not direct |
| `RandBiasFieldd` | coeff=(0, 0.05) | coeff=(0, 0.02) | Residual only in ADC |
| `RandKSpaceSpikeNoised` | prob=0.1 | prob=0.05 | Indirect effect in ADC |

---

## Resolution Scaling

Elastic deformation parameters are specified in **voxels**. For different resolutions:

```
scaled_param = base_param × (target_resolution / base_resolution)
```

Base parameters (2mm resolution):
- `sigma_range`: (3, 15)
- `magnitude_range`: (3, 10)
- `translate_range`: (0.5, 3)

---

## Dataset-Specific Considerations

### Denoised Data

If data has been preprocessed with denoising:
- Reduce `RandRicianNoised` std
- Reduce `RandBiasFieldd` coefficients (residuals only)

### Linear vs Non-linear Registration

- **Non-linear**: Anatomy standardized, less spatial augmentation needed
- **Linear**: More anatomical variability, may benefit from more spatial augmentation

---

*Last updated: 2026-01-16*
*Used by: `multimodal_base` transform dict in `transform_dicts.py`*
