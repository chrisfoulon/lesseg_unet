# Implementation Checklist

## Phase 1: Theoretical Memory Model & Hardware Detection

### File: `lesseg_unet/hardware/detection.py`
- [ ] Create `HardwareProfile` dataclass
- [ ] Implement `detect_gpus()` - VRAM, compute capability, free memory
- [ ] Implement `detect_cpu()` - cores, RAM, respect SLURM limits
- [ ] Implement `get_hardware_profile()` - main entry point
- [ ] Unit tests with mocked hardware

### File: `lesseg_unet/hardware/memory_model.py`
- [ ] Create `MemoryBreakdown` dataclass
- [ ] Implement `SwinUNETRMemoryCalculator` class:
  - [ ] `calculate_model_parameters()` - count params by layer
  - [ ] `calculate_activation_memory()` - forward pass activations
  - [ ] `calculate_gradient_memory()` - backward pass gradients
  - [ ] `calculate_optimizer_memory()` - Adam state
  - [ ] `estimate_total_memory()` - complete breakdown
  - [ ] `find_max_batch_size()` - binary search
- [ ] Implement `UNetMemoryCalculator` class (simpler version)
- [ ] Unit tests with known architectures
- [ ] Validate against actual measurements (dry-run)

### File: `lesseg_unet/hardware/validator.py`
- [ ] Implement `validate_config()` - dry-run forward + backward
- [ ] Implement `find_safe_config()` - progressive reduction on OOM
- [ ] Handle OOM gracefully (catch, reduce, retry)
- [ ] Return memory stats (peak, free)
- [ ] Integration tests

## Phase 2: Auto-Configuration Logic

### File: `lesseg_unet/auto_config/heuristics.py`
- [ ] `suggest_batch_size()` - based on VRAM and memory model
- [ ] `suggest_patch_size()` - based on image size and VRAM
- [ ] `suggest_num_workers()` - based on CPU, storage, GPU count
- [ ] `suggest_network_depth()` - 4 or 5 based on VRAM
- [ ] `suggest_feature_size()` - model capacity based on VRAM
- [ ] `allocate_resources_proportionally()` - multi-GPU CPU allocation
- [ ] Unit tests for each heuristic

### File: `lesseg_unet/auto_config/configurator.py`
- [ ] Create `AutoConfigurator` class
- [ ] Implement `suggest_config()` - orchestrate all heuristics
- [ ] Support targets: 'speed', 'memory', 'balanced'
- [ ] Apply user overrides
- [ ] Log reasoning for each decision
- [ ] Integration tests

## Phase 3: Config Management

### File: `lesseg_unet/auto_config/config_manager.py`
- [ ] Create `TrainingConfig` dataclass
- [ ] Implement `from_args()` - convert argparse to config
- [ ] Implement `save()` - write YAML, JSON, command
- [ ] Implement `load()` - read from YAML
- [ ] Implement `apply_to_args()` - reload config into argparse
- [ ] Unit tests

## Phase 4: Integration

### File: `lesseg_unet/main.py`
- [ ] Add CLI arguments:
  - [ ] `--auto_config`
  - [ ] `--num_gpus`, `--gpu_ids`
  - [ ] `--network_depth`, `--feature_size`
  - [ ] `--vram_safety_margin`, `--ram_safety_margin`
  - [ ] `--override_batch_size`, `--override_patch_size`, `--override_num_workers`
  - [ ] `--resume_from_config`
- [ ] Call `get_hardware_profile()` (always, even without auto_config)
- [ ] If `--auto_config`:
  - [ ] Analyze dataset (get median image size)
  - [ ] Call `AutoConfigurator.suggest_config()`
  - [ ] Apply to args
  - [ ] Run dry-run validation
  - [ ] Log decisions
- [ ] Create and save `TrainingConfig`
- [ ] Pass config to `training.training()`
- [ ] Integration tests

### File: `lesseg_unet/net.py`
- [ ] Modify SwinUNETR creation:
  - [ ] Accept `depths` parameter (list of ints)
  - [ ] Accept `feature_size` parameter
  - [ ] Calculate `num_heads` from depths
- [ ] Modify UNet creation (if needed):
  - [ ] Configurable channels/strides
- [ ] Unit tests

### File: `lesseg_unet/training.py`
- [ ] Accept `network_depth` parameter in `training()` signature
- [ ] Pass to model creation
- [ ] No other changes needed

## Phase 5: Testing

### Unit Tests
- [ ] Hardware detection (mock different configs)
- [ ] Memory model (test calculations against known values)
- [ ] Heuristics (test edge cases: 4GB, 24GB, 80GB)
- [ ] Config management (save/load roundtrip)

### Integration Tests
- [ ] Full auto-config pipeline (synthetic data)
- [ ] Multi-GPU resource allocation
- [ ] Dry-run validation with actual model
- [ ] OOM recovery (simulate OOM, verify fallback)

### Manual Testing
- [ ] Test on 3.7GB GPU (laptop) - should work with small config
- [ ] Test on 16GB GPU - validate predictions
- [ ] Test on 40GB GPU - validate predictions
- [ ] Compare theoretical vs actual memory usage
- [ ] Adjust safety margins if needed

## Phase 6: Documentation

- [ ] Update README with auto-config examples
- [ ] Add docstrings to all new modules
- [ ] Add user guide: when/how to use auto-config
- [ ] Add troubleshooting section
- [ ] Document limitations and edge cases

## Success Criteria

- [ ] Zero OOM errors with auto-config (tested on 3 different GPUs)
- [ ] GPU utilization >70% on average
- [ ] Theoretical model within ±10% of actual
- [ ] Multi-GPU opt-in working correctly
- [ ] Config save/load working
- [ ] All existing tests still pass
- [ ] New tests covering auto-config functionality

## Risk Mitigation

- [ ] Implement in isolated modules (minimize existing code changes)
- [ ] Make auto-config opt-in (don't break existing workflows)
- [ ] Extensive testing before enabling by default
- [ ] Document rollback procedure
- [ ] Keep existing manual configuration working

## Timeline Estimate

- Phase 1: 2-3 days (memory model + validation)
- Phase 2: 1-2 days (heuristics + configurator)
- Phase 3: 1 day (config management)
- Phase 4: 1-2 days (integration)
- Phase 5: 2 days (testing on different hardware)
- Phase 6: 1 day (documentation)

**Total: ~8-11 days** (1.5-2 weeks)
**MVP (Phases 1-3): ~4-5 days**
