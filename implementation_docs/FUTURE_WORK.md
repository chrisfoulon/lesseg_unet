# Future Work: Multi-Class Labels & Multi-Modal Controls

**Status**: Infrastructure complete, core pipeline implementation needed
**Last Updated**: 2025-12-21

---

## Overview

The CLI and data loading infrastructure has been implemented to support:
1. **Multi-class label segmentation** (e.g., lesion + edema)
2. **Multi-modal control subjects** (e.g., DWI + ADC controls)

However, these features are **not yet enabled** as they require significant changes to the training/validation pipeline that go beyond data loading.

### What Works Today ✅

- **Multi-modal images**: `-p /dwi /adc -imn dwi adc` (FULLY IMPLEMENTED)
- **Single-class labels**: `-lp /labels -lmn lesion` (FULLY IMPLEMENTED)
- **Single-modal controls**: `-ctr /controls -cmn dwi` (FULLY IMPLEMENTED)

### What's Planned But Not Implemented ⚠️

- **Multi-class labels**: `-lp /lesion /edema -lmn lesion edema` (CLI ready, pipeline not)
- **Multi-modal controls**: `-ctr /dwi /adc -cmn dwi adc` (CLI ready, pipeline not)

---

## Why Multi-Class Labels Require More Work

Multi-modal images only affect **input channels** - the model sees more input channels, but the training loop, loss, and metrics remain the same.

Multi-class labels affect **output channels AND the entire training paradigm**:

| Component | Single-Class | Multi-Class | Change Required |
|-----------|--------------|-------------|----------------|
| **Model output** | 1 channel | N channels | Auto-detected ✅ |
| **Transforms** | Load 'label' | Load 'label_lesion', 'label_edema' | How to combine? ❓ |
| **Loss function** | BCE or Dice | Per-class or combined | Major changes ❌ |
| **Metrics** | Single Dice | Per-class Dice, mean Dice | Major changes ❌ |
| **Output saving** | Single mask | Multi-channel mask | Changes needed ❌ |
| **Visualization** | Single overlay | Multi-class overlay | Changes needed ❌ |
| **Inference** | Binary prediction | Multi-class prediction | Changes needed ❌ |

---

## Implementation Roadmap

### Phase 1: Design Decisions (Required Before Implementation)

**Key Question**: Multi-label vs Multi-class?

1. **Multi-label (independent binary masks)**
   - Each class is independent (subject can have lesion AND edema)
   - Model outputs N binary channels
   - Loss: Binary cross-entropy per channel
   - Example: `output[:, 0]` = lesion probability, `output[:, 1]` = edema probability

2. **Multi-class (mutually exclusive)**
   - Classes are mutually exclusive (each voxel belongs to ONE class)
   - Model outputs N+1 channels (N classes + background)
   - Loss: Softmax + cross-entropy
   - Example: `output[:, 0]` = background, `output[:, 1]` = lesion, `output[:, 2]` = edema

**Decision needed**: Which paradigm fits your use cases?

### Phase 2: Transform Pipeline Updates

**File**: `lesseg_unet/transformations.py` or `lesseg_unet/data_utils.py`

**Current behavior**:
```python
# Single label
LoadImaged: keys=['image_dwi', 'image_adc', 'label']
ConcatItemsd: keys=['image_dwi', 'image_adc'], name='image'
# Result: 'image' (2 channels), 'label' (1 channel)
```

**Needed for multi-class**:
```python
# Multi-class labels
LoadImaged: keys=['image_dwi', 'image_adc', 'label_lesion', 'label_edema']
ConcatItemsd: keys=['image_dwi', 'image_adc'], name='image'
ConcatItemsd: keys=['label_lesion', 'label_edema'], name='label'  # NEW
# Result: 'image' (2 channels), 'label' (2 channels)
```

**Implementation**:
- Detect multi-class labels in `adapt_transforms_for_multimodal()`
- Add second `ConcatItemsd` for label keys
- Ensure channel ordering is consistent (alphabetical)

### Phase 3: Loss Function Updates

**File**: `lesseg_unet/training.py`

**Current**: Single-channel loss (e.g., `DiceLoss()` or `BCEWithLogitsLoss()`)

**Needed**:
```python
if model_config['out_channels'] > 1:
    # Multi-class or multi-label loss
    if multi_label_mode:
        # Independent binary loss per channel
        loss = sum(DiceLoss()(pred[:, i], target[:, i]) for i in range(n_classes))
    else:
        # Multi-class softmax + CE
        loss = CrossEntropyLoss()(pred, target)
else:
    # Single-class loss (current behavior)
    loss = DiceLoss()(pred, target)
```

**Considerations**:
- Class weighting for imbalanced datasets
- Combined losses (e.g., Dice + BCE per class)
- Handling missing labels per subject

### Phase 4: Metrics Updates

**File**: `lesseg_unet/training.py`

**Current**: Single Dice score

**Needed**:
```python
# Per-class metrics
metrics = {}
for i, class_name in enumerate(class_names):
    metrics[f'dice_{class_name}'] = compute_dice(pred[:, i], target[:, i])
    metrics[f'iou_{class_name}'] = compute_iou(pred[:, i], target[:, i])

# Aggregate metrics
metrics['dice_mean'] = np.mean([metrics[f'dice_{name}'] for name in class_names])
```

**Logging**:
- Log per-class metrics separately
- Log mean metrics
- Track which classes are improving/degrading

### Phase 5: Output Saving & Visualization

**File**: `lesseg_unet/training.py`, evaluation scripts

**Needed**:
- Save multi-channel predictions (NIfTI with N volumes or N separate files)
- Visualization with distinct colors per class
- Evaluation scripts that compute per-class statistics

### Phase 6: Testing & Validation

**Files**: `tests/test_transforms.py`, `tests/test_training.py` (new)

**Required tests**:
- Transform pipeline correctly concatenates multi-class labels
- Loss computation works for multi-class outputs
- Metrics computed correctly per class
- Output shapes match expected dimensions
- Backward compatibility with single-class still works

---

## Multi-Modal Controls Implementation

**Simpler than multi-class labels** - controls don't affect loss/metrics.

**Current limitation**: Controls are loaded but not integrated into training loop.

**Required changes**:
1. **Data loading**: Include control subjects in dataloaders (currently only patients)
2. **Loss masking**: Controls have no labels, skip label-based loss for them
3. **Unsupervised component**: Add unsupervised loss for controls (e.g., reconstruction, contrastive)

**File**: `lesseg_unet/training.py`

**Implementation**:
```python
for batch in dataloader:
    if 'control_dwi' in batch:  # Control subject
        # Apply unsupervised loss (e.g., autoencoder reconstruction)
        loss = reconstruction_loss(model(batch['control_dwi']), batch['control_dwi'])
    else:  # Patient subject
        # Apply supervised segmentation loss
        loss = dice_loss(model(batch['image']), batch['label'])
```

---

## Current State of Infrastructure

### Implemented ✅

1. **CLI arguments**: `-lmn`, `-cmn` accept multiple values
2. **Validation**: `validate_multimodal_arguments()` checks counts match
3. **Data loading**: `folder_mode_to_split_lists()` signature supports dicts
4. **Guards**: `NotImplementedError` raised if >1 label or >1 control used
5. **Tests**: Written but skipped (ready to enable when implemented)
6. **Model config**: `extract_model_config()` auto-detects `out_channels`

### Not Implemented ❌

1. **Transform adaptation**: Multi-class label concatenation
2. **Loss functions**: Multi-class/multi-label loss computation
3. **Metrics**: Per-class metric computation
4. **Output handling**: Multi-channel prediction saving
5. **Control integration**: Unsupervised loss for controls

---

## How to Enable When Ready

1. **Remove guards** in `main.py` (lines 531-551) and `data_utils.py` (lines 544-554)
2. **Implement Phase 2-6** from roadmap above
3. **Enable tests** by removing `@pytest.mark.skip` decorators
4. **Run full test suite** to verify backward compatibility
5. **Test with real multi-class dataset** to validate end-to-end pipeline

---

## References

- **CLI implementation**: `lesseg_unet/main.py` lines 215-241 (modality naming args)
- **Validation**: `lesseg_unet/main.py` lines 23-128 (validation functions)
- **Data loading**: `lesseg_unet/data_utils.py` lines 405-733 (`folder_mode_to_split_lists`)
- **Guards**: Search codebase for `NotImplementedError` to find all guard locations
- **Tests**: `tests/test_folder_converter.py` classes `TestMultiClassLabels`, `TestMultiModalControls`

---

## Questions to Answer Before Implementation

1. **Multi-label or multi-class paradigm?** (independent vs mutually exclusive)
2. **Loss function choice?** (Dice, BCE, CE, combined?)
3. **Class weighting strategy?** (balanced, inverse frequency, manual weights?)
4. **Missing label handling?** (some subjects may not have all classes)
5. **Output format?** (single multi-channel file vs separate files per class?)
6. **Control loss type?** (reconstruction, contrastive, adversarial?)

---

## Estimated Effort

- **Multi-class labels**: 2-3 days (design + implementation + testing)
- **Multi-modal controls**: 1-2 days (simpler, no loss/metrics changes)
- **Risk level**: Medium (isolated to training pipeline, good test coverage possible)

---

**Ready to implement when use case arises!**
