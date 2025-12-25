# Claude Code Development Notes

This file tracks ongoing development tasks and research for the lesseg_unet package.

## Current Tasks

### Deferred: Empirical Profiling for Auto-Config

**Status**: Research complete, implementation deferred

**Research Notes**: See `/tmp/empirical_profiling_research.md`

**Problem**: Auto-config's theoretical memory calculator underestimates actual GPU memory by ~2x, causing OOM errors even after gradient checkpointing.

**Proposed Solution**: Implement empirical profiling (similar to PyTorch Lightning and nnUNet) that runs test batches to measure actual memory consumption.

**Next Steps** (when resuming):
1. Implement `lesseg_unet/auto_config/profiler.py`
2. Integrate at `training.py:630-640`
3. Add comprehensive tests

---

## Recent Changes

### v2.0.8 - Gradient Checkpointing Support
- Added automatic gradient checkpointing for small GPUs (<6GB VRAM)
- New heuristics in `auto_config/heuristics.py`: `suggest_use_checkpoint()`
- Modified `SwinUNETRMemoryCalculator` to account for checkpointing (~50% memory reduction)
- All 79 tests passing
- Commit: 89c9c59

### v2.0.9 - Dependency and Installation Fixes (In Progress)

**Issue 1: Missing psutil dependency**
- **Fix**: Added `psutil>=5.9.0` to `setup.py` install_requires (line 38)
- **Why**: Required by `hardware/detection.py` for CPU/RAM detection

**Issue 2: CUDA errors despite nvidia-smi working**
- **Root Cause**: `pip install -e .` installs CPU-only PyTorch by default because setup.py doesn't specify CUDA index URL
- **Fix**: Updated README.md with proper PyTorch installation instructions (install PyTorch with CUDA BEFORE `pip install -e .`)
- **Diagnostics**: Created `/tmp/cuda_diagnostics.md` with troubleshooting guide
- **Key Commands**:
  ```bash
  # Check if you have CPU-only version
  python -c "import torch; print(torch.__version__)"
  # If shows "2.7.0+cpu", reinstall with:
  pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
  ```

**Files Modified**:
- `setup.py`: Added psutil dependency
- `README.md`: Rewrote installation section with PyTorch CUDA instructions
- Created: `/tmp/cuda_diagnostics.md`
