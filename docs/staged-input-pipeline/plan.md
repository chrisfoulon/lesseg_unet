# Implementation Plan: Staged Input Pipeline Refactoring

## Task Complexity Assessment

**Task Complexity**: MEDIUM

**Implementation Approach**: Test-driven development with incremental extraction. Each stage function will be:
1. Tested independently with unit tests
2. Extracted from existing code (low risk)
3. Integrated and verified with integration tests
4. Validated against real data for backward compatibility

**Key Challenges**:
- Maintaining exact shuffle/split behavior for reproducibility
- Ensuring all routing paths work correctly (training vs validation vs segmentation)
- Handling multi-modal edge cases (missing modalities, incomplete subjects)

**Resource Requirements**:
- **Time**: ~6-8 hours implementation + testing
- **Dependencies**: None (all existing packages)
- **Data**: Test dataset (10 subjects) + user's real DWI+ADC data for validation

---

## Progress Tracking Protocol

**CRITICAL**: After completing any task:
1. Mark checkbox in this plan: `- [x]`
2. Update TodoWrite with new status
3. Run related tests to verify
4. Update both plan.md AND TodoWrite (keep in sync!)

**Progress Indicators**:
- S = Small (< 1 hour)
- M = Medium (1-2 hours)
- L = Large (2+ hours)

---

## Hierarchical Task Structure

### Phase 1: Extract Stage Functions ║ tests/staged_input/test_stages.py ║ Extract and test each stage independently ║ L

- [ ] **1.1: Extract match_modalities_by_subject() from folder_mode_to_split_lists()** ║ M
  - [ ] 1.1.1: Copy lines 603-733 from data_utils.py to new function
  - [ ] 1.1.2: Add NumPy-style docstring with examples
  - [ ] 1.1.3: Parameterize: image_modalities, label_classes, control_modalities, require_all
  - [ ] 1.1.4: Return (subject_dicts, control_dicts) tuple
  - [ ] 1.1.5: Run flake8 on function (max-complexity 10)

- [ ] **1.2: Write tests for match_modalities_by_subject()** ║ tests/staged_input/test_match.py ║ M
  - [ ] 1.2.1: test_match_basic_two_modalities() - DWI + ADC + label
  - [ ] 1.2.2: test_match_incomplete_subject() - Missing modality raises ValueError
  - [ ] 1.2.3: test_match_no_labels() - Images only (for segmentation)
  - [ ] 1.2.4: test_match_with_controls() - Separate control dicts
  - [ ] 1.2.5: test_match_sorting_reproducible() - Subject IDs are sorted
  - [ ] 1.2.6: Run tests: `pytest tests/staged_input/test_match.py -v`
  - [ ] 1.2.7: Check coverage: ≥90%

- [ ] **1.3: Extract shuffle_and_split_subjects()** ║ M
  - [ ] 1.3.1: Copy lines 735-743 from data_utils.py to new function
  - [ ] 1.3.2: Add NumPy-style docstring
  - [ ] 1.3.3: Parameterize: subject_dicts, n_folds, shuffle, random_seed
  - [ ] 1.3.4: Add shuffle parameter (default=True)
  - [ ] 1.3.5: Return split_lists: [[fold0], [fold1], ...]
  - [ ] 1.3.6: Run flake8 on function

- [ ] **1.4: Write tests for shuffle_and_split_subjects()** ║ tests/staged_input/test_split.py ║ M
  - [ ] 1.4.1: test_shuffle_reproducible() - Same seed = same splits
  - [ ] 1.4.2: test_no_shuffle() - shuffle=False preserves order
  - [ ] 1.4.3: test_even_distribution() - Folds differ by ≤1 subject
  - [ ] 1.4.4: test_single_fold() - n_folds=1 returns [all_subjects]
  - [ ] 1.4.5: Run tests: `pytest tests/staged_input/test_split.py -v`
  - [ ] 1.4.6: Check coverage: ≥90%

- [ ] **1.5: Create list_nifti_from_folders() wrapper** ║ S
  - [ ] 1.5.1: Create function that calls _build_subject_to_file_mapping() for each folder
  - [ ] 1.5.2: Add NumPy-style docstring
  - [ ] 1.5.3: Parameterize: folders (Dict[str, Path]), pattern (str)
  - [ ] 1.5.4: Return {modality: {subject_id: path}}
  - [ ] 1.5.5: Run flake8 on function

- [ ] **1.6: Write tests for list_nifti_from_folders()** ║ tests/staged_input/test_list.py ║ M
  - [ ] 1.6.1: test_list_basic() - Lists files from multiple folders
  - [ ] 1.6.2: test_list_empty_folder() - Returns empty dict for empty folder
  - [ ] 1.6.3: test_list_missing_folder() - Raises appropriate error
  - [ ] 1.6.4: test_list_pattern_mismatch() - No files match pattern
  - [ ] 1.6.5: Run tests: `pytest tests/staged_input/test_list.py -v`
  - [ ] 1.6.6: Check coverage: ≥90%

- [ ] **1.7: Create validate_subject_dicts()** ║ utils.py ║ S
  - [ ] 1.7.1: Create function to validate all paths exist
  - [ ] 1.7.2: Add NumPy-style docstring
  - [ ] 1.7.3: Parameterize: subject_dicts, check_loadable, min_size
  - [ ] 1.7.4: Raise ValueError with clear message if invalid
  - [ ] 1.7.5: Run flake8 on function

- [ ] **1.8: Write tests for validate_subject_dicts()** ║ tests/staged_input/test_validate.py ║ S
  - [ ] 1.8.1: test_validate_success() - All paths valid
  - [ ] 1.8.2: test_validate_missing_file() - Detects missing file
  - [ ] 1.8.3: test_validate_small_file() - Detects file < min_size
  - [ ] 1.8.4: test_validate_not_loadable() - Detects corrupt NIfTI (if check_loadable=True)
  - [ ] 1.8.5: Run tests: `pytest tests/staged_input/test_validate.py -v`
  - [ ] 1.8.6: Check coverage: ≥90%

---

### Phase 2: Refactor folder_mode_to_split_lists() ║ tests/test_data_utils.py ║ Refactor to call extracted functions ║ M

- [ ] **2.1: Refactor folder_mode_to_split_lists() to use extracted functions** ║ M
  - [ ] 2.1.1: Replace lines 571-601 with call to list_nifti_from_folders()
  - [ ] 2.1.2: Replace lines 603-733 with call to match_modalities_by_subject()
  - [ ] 2.1.3: Replace lines 735-743 with call to shuffle_and_split_subjects()
  - [ ] 2.1.4: Verify function signature unchanged (backward compatibility)
  - [ ] 2.1.5: Update docstring if needed
  - [ ] 2.1.6: Run flake8 on function

- [ ] **2.2: Verify backward compatibility** ║ M
  - [ ] 2.2.1: Run existing tests: `pytest tests/test_data_utils.py -v`
  - [ ] 2.2.2: Run existing tests: `pytest tests/test_folder_converter.py -v`
  - [ ] 2.2.3: All existing tests must pass unchanged
  - [ ] 2.2.4: Create test_folder_mode_unchanged() - Verify same output as before

- [ ] **2.3: Test with real data** ║ M
  - [ ] 2.3.1: Use user's test dataset (57 subjects, DWI+ADC+label)
  - [ ] 2.3.2: Generate split_lists.json with refactored function
  - [ ] 2.3.3: Compare with previous split_lists.json (should be identical)
  - [ ] 2.3.4: Verify shuffle reproducibility (seed=42)

---

### Phase 3: Update main.py Routing ║ tests/test_cli_validation.py ║ Add training vs inference detection ║ L

- [ ] **3.1: Add is_training detection in main.py** ║ S
  - [ ] 3.1.1: Add line after arg parsing: `is_training = args.checkpoint is None`
  - [ ] 3.1.2: Document logic in comment

- [ ] **3.2: Split input handling for training vs inference** ║ M
  - [ ] 3.2.1: Wrap folder_mode_to_split_lists() call in `if is_training:` block
  - [ ] 3.2.2: Add `else:` block for inference:
    - Call list_nifti_from_folders()
    - Call match_modalities_by_subject()
    - Validate with validate_subject_dicts()
    - Set img_list = subject_dicts (flat list)
  - [ ] 3.2.3: Set les_list = None in both branches (labels embedded)
  - [ ] 3.2.4: Run flake8 on modified section

- [ ] **3.3: Update label detection routing** ║ S
  - [ ] 3.3.1: Simplify lines 1221-1232 (remove nested list handling)
  - [ ] 3.3.2: For inference, img_list is flat list, check first_item directly
  - [ ] 3.3.3: Detect has_labels: any(k.startswith('label_') for k in first_item.keys())

- [ ] **3.4: Add --subject-list CLI argument** ║ S
  - [ ] 3.4.1: Add argument: `--subject-list` (path to pre-matched JSON)
  - [ ] 3.4.2: Add parsing logic similar to --pretrained-split-list
  - [ ] 3.4.3: For training: load → validate → shuffle_and_split_subjects()
  - [ ] 3.4.4: For inference: load → validate → use directly

- [ ] **3.5: Write tests for main.py routing** ║ tests/staged_input/test_routing.py ║ M
  - [ ] 3.5.1: test_training_mode_detection() - checkpoint=None → is_training=True
  - [ ] 3.5.2: test_inference_mode_detection() - checkpoint set → is_training=False
  - [ ] 3.5.3: test_training_calls_split() - Verify folder_mode_to_split_lists called
  - [ ] 3.5.4: test_inference_no_split() - Verify NO shuffle/split for inference
  - [ ] 3.5.5: test_label_detection_flat_list() - Detects labels in flat list
  - [ ] 3.5.6: Run tests: `pytest tests/staged_input/test_routing.py -v`

---

### Phase 4: Create Split-Lists-Aware Loops ║ tests/staged_input/test_loops.py ║ Handle dict format in validation/segmentation ║ L

- [ ] **4.1: Create validation_loop_split_lists()** ║ segmentation.py ║ M
  - [ ] 4.1.1: Copy validation_loop() as starting point
  - [ ] 4.1.2: Modify to accept flat list of dicts (not separate img/seg lists)
  - [ ] 4.1.3: Add split_lists wrapping: `split_lists = [subject_dicts]`
  - [ ] 4.1.4: Call adapt_transforms_for_multimodal(transform_dict, split_lists)
  - [ ] 4.1.5: Create dataset: `Dataset(subject_dicts, transform=val_transforms)`
  - [ ] 4.1.6: Keep rest of validation logic unchanged
  - [ ] 4.1.7: Add NumPy-style docstring
  - [ ] 4.1.8: Run flake8 on function
  - [ ] 4.1.9: **Fix F821 error (line 241: undefined vol_output)** while working here

- [ ] **4.2: Create segmentation_loop_split_lists()** ║ segmentation.py ║ M
  - [ ] 4.2.1: Copy segmentation_loop() as starting point
  - [ ] 4.2.2: Modify to accept flat list of dicts (not path list)
  - [ ] 4.2.3: Add split_lists wrapping: `split_lists = [subject_dicts]`
  - [ ] 4.2.4: Call adapt_transforms_for_multimodal(transform_dict, split_lists)
  - [ ] 4.2.5: Create dataset with image-only transforms
  - [ ] 4.2.6: Keep rest of segmentation logic unchanged
  - [ ] 4.2.7: Add NumPy-style docstring
  - [ ] 4.2.8: Run flake8 on function

- [ ] **4.3: Update main.py to call new loop functions** ║ S
  - [ ] 4.3.1: Replace validation_loop() call with validation_loop_split_lists()
  - [ ] 4.3.2: Replace segmentation_loop() call with segmentation_loop_split_lists()
  - [ ] 4.3.3: Pass flat img_list (not nested split_lists)
  - [ ] 4.3.4: Update function signatures/parameters as needed

- [ ] **4.4: Write tests for loop functions** ║ tests/staged_input/test_loops.py ║ L
  - [ ] 4.4.1: test_validation_loop_dict_format() - Accepts flat dict list
  - [ ] 4.4.2: test_validation_loop_transforms_adapted() - Transforms adapted correctly
  - [ ] 4.4.3: test_segmentation_loop_dict_format() - Accepts flat dict list
  - [ ] 4.4.4: test_segmentation_loop_no_labels() - Works without labels
  - [ ] 4.4.5: Create mock checkpoint and dataset for testing
  - [ ] 4.4.6: Run tests: `pytest tests/staged_input/test_loops.py -v`

---

### Phase 5: Integration Testing ║ tests/staged_input/test_integration.py ║ End-to-end pipeline tests ║ L

- [ ] **5.1: Test end-to-end training pipeline** ║ M
  - [ ] 5.1.1: Create test dataset (10 subjects, DWI+ADC+label)
  - [ ] 5.1.2: test_training_folders_to_split_lists() - Full pipeline
  - [ ] 5.1.3: Verify split_lists.json created correctly
  - [ ] 5.1.4: Verify subjects shuffled with seed=42
  - [ ] 5.1.5: Verify folds have even distribution
  - [ ] 5.1.6: Run test: `pytest tests/staged_input/test_integration.py::test_training_folders_to_split_lists -v`

- [ ] **5.2: Test end-to-end validation pipeline** ║ M
  - [ ] 5.2.1: Use same test dataset
  - [ ] 5.2.2: test_validation_folders_to_flat_list() - NO shuffle
  - [ ] 5.2.3: Verify img_list is flat list (not nested)
  - [ ] 5.2.4: Verify labels detected correctly
  - [ ] 5.2.5: Verify routing to validation_loop_split_lists()
  - [ ] 5.2.6: Run test: `pytest tests/staged_input/test_integration.py::test_validation_folders_to_flat_list -v`

- [ ] **5.3: Test end-to-end segmentation pipeline** ║ M
  - [ ] 5.3.1: Create test dataset without labels (images only)
  - [ ] 5.3.2: test_segmentation_folders_to_flat_list() - NO shuffle
  - [ ] 5.3.3: Verify img_list is flat list
  - [ ] 5.3.4: Verify NO labels detected
  - [ ] 5.3.5: Verify routing to segmentation_loop_split_lists()
  - [ ] 5.3.6: Run test: `pytest tests/staged_input/test_integration.py::test_segmentation_folders_to_flat_list -v`

- [ ] **5.4: Test backward compatibility with real data** ║ M
  - [ ] 5.4.1: Use user's actual dataset (57 test subjects, DWI+ADC+label)
  - [ ] 5.4.2: Generate split_lists.json with new code
  - [ ] 5.4.3: Compare with original split_lists.json (byte-for-byte if possible)
  - [ ] 5.4.4: Verify same shuffle order (seed=42)
  - [ ] 5.4.5: Verify resume training works (auto-loads splits)

- [ ] **5.5: Run full test suite** ║ S
  - [ ] 5.5.1: `pytest tests/ -v 2>&1 | tail -100`
  - [ ] 5.5.2: All 204 existing tests must pass
  - [ ] 5.5.3: All new tests must pass
  - [ ] 5.5.4: Coverage report: `pytest --cov=lesseg_unet --cov-report=term-missing 2>&1 | tail -150`
  - [ ] 5.5.5: Verify ≥90% coverage for new code

---

### Phase 6: Maintenance and Cleanup ║ Apply Boy Scout Rule ║ S

- [ ] **6.1: Fix high-priority maintenance issues** ║ S
  - [ ] 6.1.1: Fix segmentation.py:241 - F821 undefined vol_output (DONE in Phase 4.1.9)
  - [ ] 6.1.2: Remove main.py:824 - F811 redundant nib import
  - [ ] 6.1.3: Fix main.py:924, 987 - F841 unused variables

- [ ] **6.2: Fix medium-priority maintenance issues (optional)** ║ S
  - [ ] 6.2.1: Clean up segmentation.py unused variables (lines 62, 102, 151, 530, 564)
  - [ ] 6.2.2: Only if not breaking functionality

- [ ] **6.3: Update documentation** ║ S
  - [ ] 6.3.1: Update CLAUDE.md with implementation notes
  - [ ] 6.3.2: Add docstrings to all new functions (DONE inline)
  - [ ] 6.3.3: Update feature_vars.md with final status

---

### Phase 7: Quality Finalization ║ Final checks and commit ║ M

- [ ] **7.1: Run flake8 on all modified files** ║ S
  - [ ] 7.1.1: `flake8 lesseg_unet/data_utils.py`
  - [ ] 7.1.2: `flake8 lesseg_unet/main.py`
  - [ ] 7.1.3: `flake8 lesseg_unet/segmentation.py`
  - [ ] 7.1.4: `flake8 lesseg_unet/utils.py`
  - [ ] 7.1.5: `flake8 tests/staged_input/`
  - [ ] 7.1.6: Zero violations in new code

- [ ] **7.2: Coverage verification** ║ S
  - [ ] 7.2.1: Generate coverage report
  - [ ] 7.2.2: Verify ≥90% for all new functions
  - [ ] 7.2.3: Document any intentional gaps (e.g., error handling paths)

- [ ] **7.3: Git commit with conventional commit format** ║ S
  - [ ] 7.3.1: Stage changes: `git add lesseg_unet/data_utils.py lesseg_unet/main.py lesseg_unet/segmentation.py lesseg_unet/utils.py tests/`
  - [ ] 7.3.2: Commit: `git commit -m "feat(input): refactor to staged input pipeline..."`
  - [ ] 7.3.3: Detailed commit message:
    ```
    feat(input): refactor to staged input pipeline

    Refactor monolithic folder_mode_to_split_lists() into composable stages:
    - list_nifti_from_folders(): List files from folders
    - match_modalities_by_subject(): Match modalities by ID
    - shuffle_and_split_subjects(): Shuffle and split (training only)
    - validate_subject_dicts(): Validate paths

    Fixes:
    - Validation/segmentation no longer shuffle test data
    - Flat list format for inference (no unnecessary nesting)
    - Add --subject-list CLI for pre-matched JSON

    Maintains:
    - Backward compatibility (same shuffle/split behavior)
    - All existing tests pass
    - Resume training works unchanged

    Tests: 204 existing + 40 new tests (≥90% coverage)

    Closes: staged-input-pipeline refactoring
    ```
  - [ ] 7.3.4: Push to remote: `git push origin dev`

---

## Success Criteria Checklist

### Functional Requirements
- [ ] Training with folders: Shuffles (seed=42), creates folds, saves split_lists.json
- [ ] Training with `-psl`: Loads pre-split, validates paths, no reshuffling
- [ ] Validation with folders: NO shuffle, flat list, routes to validation
- [ ] Segmentation with folders: NO shuffle, flat list, routes to segmentation
- [ ] Resume training: Auto-loads splits, no reshuffling
- [ ] New `--subject-list`: Accepts pre-matched flat JSON

### Quality Requirements
- [ ] All 204 existing tests pass
- [ ] All new tests pass (≥40 tests)
- [ ] Coverage ≥90% for new code
- [ ] Flake8 compliant (max-complexity 10)
- [ ] NumPy-style docstrings for all new functions

### Architecture Requirements
- [ ] Each stage is separate, testable function
- [ ] Functions compose: Stage 0→1→2→3
- [ ] Controls use same functions, separate pipeline
- [ ] Clear separation: generic vs training-specific
- [ ] Backward compatible: folder_mode_to_split_lists() unchanged signature

---

## Estimated Timeline

| Phase | Description | Estimated Time |
|-------|-------------|---------------|
| 1 | Extract stage functions + tests | 2-3 hours |
| 2 | Refactor folder_mode_to_split_lists | 1 hour |
| 3 | Update main.py routing | 1-2 hours |
| 4 | Create split-lists-aware loops | 2 hours |
| 5 | Integration testing | 1-2 hours |
| 6 | Maintenance and cleanup | 30 min |
| 7 | Quality finalization | 30 min |
| **Total** | **Complete implementation** | **8-11 hours** |

---

## Next Actions

1. **Start Phase 1**: Extract and test `match_modalities_by_subject()`
2. **Update TodoWrite**: Mark Phase 1.1 as in_progress
3. **Create test file**: `tests/staged_input/test_match.py`
4. **Write first test**: test_match_basic_two_modalities()
5. **Implement function**: Extract from data_utils.py
6. **Verify**: Run test, achieve ≥90% coverage

**Command to start**:
```bash
mkdir -p tests/staged_input
touch tests/staged_input/__init__.py
touch tests/staged_input/test_match.py
```
