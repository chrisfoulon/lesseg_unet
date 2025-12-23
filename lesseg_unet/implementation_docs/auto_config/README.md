# Auto-Configuration Implementation Documentation

This folder contains implementation documentation for the hardware-aware auto-configuration feature.

## Files

1. **00_OVERVIEW.md** - Feature overview, goals, architecture
2. **01_CONTEXT.md** - Existing code context and integration points
3. **02_API_DESIGN.md** - Public API, CLI arguments, examples
4. **03_MEMORY_MODEL.md** - Theoretical memory calculation formulas
5. **04_IMPLEMENTATION_CHECKLIST.md** - Step-by-step tasks

## Quick Start

1. Read `00_OVERVIEW.md` for high-level understanding
2. Check `01_CONTEXT.md` for existing code integration points
3. Review `03_MEMORY_MODEL.md` for memory calculation approach
4. Follow `04_IMPLEMENTATION_CHECKLIST.md` for implementation

## Key Principles

- OOM prevention is paramount (never cause VRAM/RAM OOM)
- Theoretical model (no empirical profiling on limited hardware)
- Minimal code impact (isolated modules)
- Opt-in multi-GPU (never grab all resources)
- Full reproducibility (save complete config)

## Implementation Status

See `04_IMPLEMENTATION_CHECKLIST.md` for current progress.

## Context Optimization Command

To optimize context for implementation, use:
```
/compact <instructions>
```

Where `<instructions>` is a summary of:
- Current task (e.g., "Implement SwinUNETRMemoryCalculator")
- Relevant files (e.g., "lesseg_unet/hardware/memory_model.py")
- Key constraints (e.g., "Use theoretical calculation, no empirical profiling")
