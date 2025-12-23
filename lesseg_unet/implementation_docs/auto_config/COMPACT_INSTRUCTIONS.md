# Context Optimization Instructions for Auto-Config Implementation

## Summary

Implement hardware-aware auto-configuration for lesseg_unet to prevent OOM errors and maximize training efficiency.

## Current State

✅ **Completed**: Pattern-based transform key expansion (label_stroke, image_dwi, etc.)
🚧 **Next**: Auto-config feature (hardware detection + memory model + config management)

## Essential Information

### Key Requirements
1. **OOM Prevention**: Use theoretical memory model (not empirical profiling)
2. **Multi-GPU**: Opt-in only, proportional CPU allocation (Y/X GPUs = Y/X CPU cores)
3. **Network Depth**: Configurable 4 or 5 layers (affects memory and patch size granularity)
4. **Defaults**: 95% VRAM usage, 100% GPU utilization target (dedicated machines)
5. **Config Storage**: Save complete resolved config (YAML/JSON) for reproducibility

### Architecture
```
lesseg_unet/
├── hardware/
│   ├── detection.py       # GPU/CPU/RAM detection
│   ├── memory_model.py    # Theoretical VRAM calculation
│   └── validator.py       # Dry-run testing
├── auto_config/
│   ├── configurator.py    # Main orchestrator
│   ├── heuristics.py      # Batch size, patch size rules
│   └── config_manager.py  # Save/load configs
├── implementation_docs/auto_config/  # Documentation (this folder)
└── [minimal changes to main.py, net.py, training.py]
```

### Integration Points
- **main.py**: Add CLI args, call auto-config, save config
- **net.py**: Accept `depths` param for SwinUNETR
- **training.py**: Accept `network_depth` param
- **data_loading.py**: No changes (already parameterized)

### Memory Calculation (Theoretical)
```
Total VRAM = Model_Params + Optimizer + Activations + Gradients + Overhead + Fragmentation

Where:
- Model_Params: Count parameters × 4 bytes (FP32)
- Optimizer: Model_Params × 2 (Adam: momentum + variance)
- Activations: batch_size × layer_outputs (depends on architecture)
- Gradients: ≈ Activations (backprop storage)
- Overhead: ~400 MB (PyTorch CUDA context)
- Fragmentation: 8% of allocated memory
```

### Key Files to Reference
- Implementation docs: `lesseg_unet/implementation_docs/auto_config/`
- Memory formulas: `03_MEMORY_MODEL.md`
- Integration points: `01_CONTEXT.md`
- API design: `02_API_DESIGN.md`
- Checklist: `04_IMPLEMENTATION_CHECKLIST.md`

### User Hardware
- **Development**: 3.7GB VRAM laptop (limited, use for testing minimal configs)
- **Validation**: 16GB and 40GB GPUs available (validate theoretical model predictions)

### Focus
- **Primary model**: SwinUNETR (most resource-hungry)
- **Secondary**: UNet (simpler, same principles)

## Implementation Strategy

**Phase 1** (Start here):
1. Implement `hardware/detection.py` (GPU/CPU/RAM detection)
2. Implement `hardware/memory_model.py` (SwinUNETRMemoryCalculator)
3. Implement `hardware/validator.py` (dry-run testing)

**Phase 2**:
4. Implement `auto_config/heuristics.py` (batch size, patch size rules)
5. Implement `auto_config/configurator.py` (orchestrate heuristics)

**Phase 3**:
6. Implement `auto_config/config_manager.py` (save/load YAML/JSON)
7. Integrate into `main.py` (CLI args, call auto-config)

## Testing Hardware Available
- 3.7GB GPU (laptop) - develop & test minimal configs
- 16GB GPU - validate predictions
- 40GB GPU - validate predictions

## Success Criteria
- Zero OOM errors with auto-config
- Theoretical model within ±10% of actual
- Multi-GPU proportional resource allocation working
- Config save/load working
- All existing tests pass

## Open Questions
None - plan approved, ready to implement.

## Commands to Remember
- Read docs: `lesseg_unet/implementation_docs/auto_config/*.md`
- Run tests: `pytest tests/`
- Check git status: `git status`
- Current branch: `dev` (last commit: label expansion feature)

## What NOT to Include in Context
- Full conversation history
- Exploratory discussions
- Verbose explanations
- Duplicate information
- Temporary /tmp notes (consolidated into implementation_docs/)
