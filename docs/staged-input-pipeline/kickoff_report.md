# Phase 0: Feature Kickoff Report
## Staged Input Pipeline Refactoring

**Date**: 2026-01-13
**Feature Slug**: `staged-input-pipeline`
**Project**: lesseg_unet

---

## 1. Environment Status

### Python Environment
- **Python Version**: 3.12.12 ✅ (≥3.11 required)
- **Virtual Environment**: Active
- **Package Manager**: pip

### Git Repository
- **Status**: Clean working directory except:
  - Modified: `lesseg_unet/main.py` (from previous bug fixes)
- **Branch**: dev
- **Repository**: Initialized and functional

### LAD Framework
- **Status**: ✅ Complete and properly structured
- **Location**: `.lad/claude_prompts/`
- **Prompts**: All 18 framework files present
- **Integrity**: Framework validated, will not be modified

---

## 2. Quality Configuration

### Created Configuration Files

#### `.flake8` ✅
```ini
[flake8]
max-line-length = 88
max-complexity = 10
ignore = E203, E266, E501, W503
exclude = .git,__pycache__,docs/,build/,dist/,.lad/,venv/,env/
```

#### `.coveragerc` ✅
```ini
[run]
branch = True
source = lesseg_unet
omit = */tests/*, */test_*, */__pycache__/*, */.*,  .lad/*, setup.py, */venv/*, */env/*

[report]
show_missing = True
skip_covered = False

[html]
directory = coverage_html
```

#### `pytest.ini` ✅
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = --strict-markers --strict-config
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
```

---

## 3. Baseline Quality Metrics

### Test Suite Baseline
- **Total Tests**: 204 tests collected
- **Test Files**: 10 test modules in `tests/` directory
- **Test Status**: 19 passed, 1 error (unrelated to feature)
  - Error: `AutoConfigResult.__init__()` missing parameter (existing issue)
- **Test Discovery**: Functional

### Code Quality Baseline (Flake8)
**Total Violations**: 260 (existing codebase)

**Breakdown by Category**:
- **C901 (Complexity)**: 3 functions exceed limit (14-16 complexity)
- **F401 (Unused imports)**: 65 occurrences
- **F541 (Empty f-strings)**: 31 occurrences
- **E124 (Indentation)**: 56 occurrences
- **F821 (Undefined names)**: 1 occurrence ⚠️ (potential bug)
- **F841 (Unused variables)**: 8 occurrences
- **Other style issues**: ~95 occurrences

**Note**: Baseline violations are existing technical debt. New code will be flake8-compliant.

### Coverage Baseline
**Not established** - Will measure after first test implementation

---

## 4. Feature Preparation

### Documentation Structure Created ✅
```
docs/staged-input-pipeline/
├── feature_vars.md          # Feature requirements and variables
├── kickoff_report.md        # This file
├── context.md               # (Phase 1) Codebase exploration
└── plan.md                  # (Phase 1) Implementation plan
```

### Feature Variables Saved ✅
**Location**: `docs/staged-input-pipeline/feature_vars.md`

**Key Variables**:
```bash
FEATURE_SLUG="staged-input-pipeline"
PROJECT_NAME="lesseg_unet"
FEATURE_DESCRIPTION="Refactor input handling into modular stages"
```

### Quality Standards for This Feature

#### Code Quality
- **Max Complexity**: 10 per function (enforced by flake8)
- **Docstring Style**: NumPy-style required
- **Line Length**: 88 characters max
- **Imports**: Sorted and used

#### Testing Standards
- **Coverage Target**: ≥90% for new code
- **Test Strategy**: TDD approach, stage-by-stage
- **Test Types**: Unit tests for each stage function
- **Test Organization**: `tests/staged_input/test_<stage>.py`

#### Commit Standards
- **Format**: Conventional commits
- **Examples**:
  - `feat(input): add stage 1 path mapping function`
  - `test(input): add tests for match_modalities_by_subject`
  - `refactor(input): extract generic matching from folder_mode_to_split_lists`

---

## 5. Problem Context

### Current Issue
The input handling in `main.py` always calls `folder_mode_to_split_lists()` which:
1. **Always shuffles** data (seed=42) - Inappropriate for validation/segmentation
2. **Always creates folds** - Unnecessary for inference (creates nested [[dict], [dict]])
3. **Combines responsibilities** - Listing, matching, validating, splitting in one function

### Impact
- Test data is shuffled → Non-deterministic validation results
- Validation/segmentation create unnecessary nested structures → Code complexity
- Poor modularity → Cannot reuse matching logic for different purposes
- Hard to extend → Adding new input formats requires modifying monolithic function

### Root Cause Analysis
**file**: `lesseg_unet/data_utils.py`
**function**: `folder_mode_to_split_lists()` (lines 443-744)
**problem**: Lines 735-743 always execute (shuffle + split), regardless of use case

### Proposed Solution Architecture
Extract into composable stages:
- **Stage 0→1**: List files from various sources (folders, CSVs, JSON)
- **Stage 1→2**: Match modalities by subject ID (generic, reusable)
- **Stage 2→3**: Shuffle and split (training-only operation)

**Benefits**:
- Single Responsibility Principle (each function does one thing)
- Composability (can enter pipeline at any stage)
- Testability (each stage is 20-30 lines, trivial to test)
- Extensibility (add new input source = one new function)

---

## 6. Related Work Completed

### Recent Bug Fixes (Same Session)
1. **Resume data leakage fix** (commit 6f5469c)
   - Auto-loads `split_lists.json` when resuming to prevent re-shuffling
   - Lines affected: `main.py:1111-1151`

2. **Sorting fix for reproducible splits** (commit d5dc618)
   - Added `sorted()` to subject IDs before shuffling
   - Lines affected: `data_utils.py:707, 725`

3. **Early stopping restoration** (commit 1fba2d3)
   - Reads tensorboard events to restore best_metric_epoch on resume
   - New function: `utils.get_best_epoch_from_events()`

### Analysis Documents Created
- `/tmp/split_lists_validation_analysis.md` - Technical analysis (code mapping, existing functions)
- `/tmp/architecture_verdict.md` - Verdict on proposed architecture
- `/tmp/staged_input_pipeline.md` - Detailed staged architecture design
- `/tmp/input_pipeline_analysis.md` - Current code mapping and proposed solution

---

## 7. Quality Gates Status

- ✅ **Required configuration files exist and are valid**
  - `.flake8`, `.coveragerc`, `pytest.ini` created
- ✅ **Development environment is functional**
  - Python 3.12.12, git repository, LAD framework
- ✅ **Baseline metrics are established**
  - 204 tests, 260 flake8 violations (existing)
- ✅ **Feature documentation structure is prepared**
  - `docs/staged-input-pipeline/` with feature_vars.md
- ✅ **Quality standards are defined and measurable**
  - Coverage ≥90%, max-complexity 10, NumPy docstrings

---

## 8. Integration Context (From Previous Analysis)

### Files That Will Be Modified
1. **lesseg_unet/data_utils.py** (~150 lines)
   - Extract `match_modalities_by_subject()` (generic matching)
   - Extract `shuffle_and_split_subjects()` (training-only)
   - Add `list_nifti_from_folders()` wrapper
   - Refactor `folder_mode_to_split_lists()` to call extracted functions

2. **lesseg_unet/utils.py** (~30 lines)
   - Add `validate_subject_dicts()` function
   - Add `load_path_lists_from_files()` function

3. **lesseg_unet/main.py** (~100 lines)
   - Add `is_training = args.checkpoint is None` detection
   - Split input handling: training vs inference branches
   - Update routing logic (check label keys, not `les_list` variable)
   - Add `--subject-list` CLI argument

4. **lesseg_unet/segmentation.py** (~200 lines)
   - Create `validation_loop_split_lists()` (handles dicts with labels)
   - Create `segmentation_loop_split_lists()` (handles dicts without labels)

### Files That Will NOT Be Modified
- ✅ `lesseg_unet/training.py` - Already handles both split_lists and flat lists
- ✅ `lesseg_unet/data_loading.py` - Already handles dicts via Dataset
- ✅ `lesseg_unet/transformations.py` - Already adapted for multi-modal

### Dependencies (Existing Functions to Reuse)
- `_build_subject_to_file_mapping()` (data_utils.py) - Lists NIfTI files
- `file_to_list()` (utils.py) - Loads CSV/text lists
- `check_inputs()` (utils.py) - Validates paths
- `adapt_transforms_for_multimodal()` (data_utils.py) - Adapts transforms
- `create_dataset()` (data_loading.py) - Creates MONAI datasets

---

## 9. Risks and Constraints

### Constraints
- **Backward compatibility**: All existing training runs must work unchanged
- **No checkpoint format changes**: Resume training must continue working
- **Preserve CLI**: All current flags must continue working
- **Zero functional changes**: Same shuffle (seed=42), same split behavior

### Risks
- **Risk**: Breaking existing training workflows
  - **Mitigation**: Comprehensive tests, verify split_lists.json unchanged
- **Risk**: Regression in multi-modal matching
  - **Mitigation**: Test with existing test data (DWI+ADC datasets)
- **Risk**: Validation/segmentation routing errors
  - **Mitigation**: Test all routing paths (labels vs no labels)

---

## 10. Success Criteria

### Functional Requirements
- [ ] Training with folders: Shuffles, creates folds, saves split_lists.json
- [ ] Training with `-psl`: Loads pre-split, validates, no reshuffling
- [ ] Validation with folders: NO shuffle, flat list, routes correctly
- [ ] Segmentation with folders: NO shuffle, flat list, routes correctly
- [ ] Resume training: Auto-loads splits, no reshuffling
- [ ] New `--subject-list`: Accepts pre-matched flat JSON

### Quality Requirements
- [ ] All 204 existing tests pass
- [ ] New tests for each stage (≥90% coverage)
- [ ] Flake8 compliant (max-complexity 10)
- [ ] NumPy-style docstrings

### Architecture Requirements
- [ ] Each stage is separate, testable function
- [ ] Functions compose: Stage 0→1→2→3
- [ ] Controls use same functions, separate pipeline
- [ ] Clear separation: generic vs training-specific

---

## 11. Next Steps

### Phase 1: Autonomous Context Planning
**Prompt**: `.lad/claude_prompts/01_autonomous_context_planning.md`

**Tasks**:
1. Autonomous codebase exploration using Task tool
2. Document integration points in `context.md`
3. Identify maintenance opportunities in target files
4. Create detailed TDD plan in `plan.md`
5. Use TodoWrite for progress tracking

**Expected Deliverables**:
- `docs/staged-input-pipeline/context.md` - Multi-level codebase understanding
- `docs/staged-input-pipeline/plan.md` - Hierarchical TDD implementation plan
- TodoWrite task list initialized

### Phase 2: Iterative Implementation
Execute TDD plan stage-by-stage with continuous testing

### Phase 3: Quality Finalization
Review, validate, and finalize implementation

---

## 12. Environment Limitations & Notes

### Known Issues
- 1 existing test failure: `test_config_manager.py::TestFromAutoConfig::test_from_auto_config_minimal`
  - Cause: `AutoConfigResult.__init__()` missing `use_checkpoint` parameter
  - Impact: None on this feature (unrelated module)
  - Action: Document but do not fix (out of scope)

### Development Tools Available
- pytest ✅
- flake8 ✅
- coverage ✅
- git ✅
- Python 3.12.12 ✅

---

**Status**: ✅ Environment ready for Phase 1 (Context Planning)
**Quality Gates**: ✅ All passed
**Next Action**: Proceed to autonomous context planning
