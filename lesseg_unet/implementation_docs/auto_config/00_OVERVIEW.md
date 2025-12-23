# Auto-Configuration Feature Implementation

## Goal
Automatically configure training parameters (batch size, patch size, num_workers, network depth) based on available hardware to prevent OOM errors and maximize efficiency.

## Key Requirements
1. **OOM Prevention**: NEVER cause VRAM or RAM OOM (paramount)
2. **Theoretical Memory Model**: Calculate memory usage from architecture (no empirical profiling needed)
3. **Multi-GPU Opt-In**: Default 1 GPU, proportional CPU allocation when multi-GPU requested
4. **95% VRAM Default**: Maximum efficiency for dedicated machines (configurable)
5. **Minimal Code Impact**: Isolated modules, clear integration points
6. **Full Reproducibility**: Save complete resolved config (not just command)

## Architecture
```
lesseg_unet/
├── hardware/
│   ├── detection.py          # Detect GPU, CPU, RAM
│   ├── memory_model.py        # Theoretical VRAM calculation
│   └── validator.py           # Dry-run config testing
├── auto_config/
│   ├── configurator.py        # Main auto-config logic
│   ├── heuristics.py          # Batch size, patch size rules
│   └── config_manager.py      # Save/load complete configs
└── implementation_docs/auto_config/  # THIS FOLDER
```

## Implementation Phases
1. **Theoretical Memory Model** (hardware/memory_model.py)
2. **Hardware Detection** (hardware/detection.py)
3. **Configuration Heuristics** (auto_config/)
4. **Dry-Run Validation** (hardware/validator.py)
5. **Integration** (main.py, training.py)
6. **Config Management** (auto_config/config_manager.py)

## User Decisions
- Multi-GPU: Opt-in, proportional resources
- Network depth: 4 or 5 layers (auto-configured)
- Thresholds: 95% VRAM default (user configurable)
- Focus: SwinUNETR primarily, UNet secondary
- Testing: Validate on 16GB and 40GB GPUs

## Files in This Folder
- `00_OVERVIEW.md` - This file (project overview)
- `01_CONTEXT.md` - Existing code context and integration points
- `02_API_DESIGN.md` - Public API and CLI arguments
- `03_MEMORY_MODEL.md` - Theoretical memory calculation formulas
- `04_IMPLEMENTATION_CHECKLIST.md` - Step-by-step implementation tasks
