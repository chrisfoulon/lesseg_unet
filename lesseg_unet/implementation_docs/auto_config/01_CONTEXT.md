# Existing Code Context

## Key Integration Points

### 1. main.py (Argument Parsing & Training Invocation)
**Location**: `lesseg_unet/main.py`
**Current state**: Uses argparse, calls `training.training()` function
**Integration needed**:
- Add new CLI arguments (--auto_config, --num_gpus, --network_depth, etc.)
- Call hardware detection
- Call auto-configurator
- Save resolved config
- Pass config to training function

**Existing relevant args**:
```python
batch_size: int = 1          # Line ~157 in training()
dataloader_workers: int = 4
model_type: str              # 'unet', 'swinunetr', 'segresnet'
```

### 2. training.py (Model Creation & Training Loop)
**Location**: `lesseg_unet/training.py`
**Function**: `training()` signature at line 147
**Current model creation**: Calls functions in `net.py`
**Integration needed**:
- Accept `network_depth` parameter
- Pass to model creation functions
- No other changes needed

**Existing parameters to respect**:
```python
def training(
    img_path_list,
    lesion_path_list,
    batch_size=1,              # ← Override with auto-config
    dataloader_workers=4,      # ← Override with auto-config
    # ... many other params
)
```

### 3. net.py (Model Creation)
**Location**: `lesseg_unet/net.py`
**Functions**: Model creation for different architectures
**Integration needed**:
- Modify SwinUNETR creation to accept `depths` parameter
- Modify UNet creation to accept configurable channels/strides

**Current SwinUNETR creation** (approximate):
```python
def create_swinunetr(in_channels, out_channels, feature_size=48, ...):
    model = SwinUNETR(
        img_size=patch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        # depths is hardcoded or has default
    )
    return model
```

### 4. data_loading.py (DataLoader Creation)
**Location**: `lesseg_unet/data_loading.py`
**Functions**:
- `create_training_data_loader()` (line 76)
- `create_fold_dataloaders()` (line 176)

**Integration needed**:
- None (already accepts batch_size and num_workers)
- Just pass auto-configured values

### 5. Transform Dictionaries
**Location**: `lesseg_unet/data/transform_dicts.py`
**Current state**: Define patch sizes statically
**Integration needed**:
- Dynamic patch size in transform dict
- Or: Generate transform dict with auto-configured patch size

**Example**:
```python
p96 = {
    'patches': [{'RandCropByPosNegLabeld': {
        'spatial_size': [96, 96, 96],  # ← Make configurable
        ...
    }}]
}
```

## Current Workflow
```
main.py
  ↓
  parse_args()
  ↓
  training.training(
      batch_size=args.batch_size,
      dataloader_workers=args.dataloader_workers,
      ...
  )
  ↓
  create model (net.py)
  ↓
  create dataloader (data_loading.py)
  ↓
  training loop
```

## Auto-Config Workflow
```
main.py
  ↓
  parse_args()
  ↓
  detect_hardware()                    # NEW
  ↓
  if args.auto_config:
      auto_config = suggest_config()   # NEW
      args.batch_size = auto_config['batch_size']
      args.dataloader_workers = auto_config['num_workers']
      # ...
  ↓
  save_config()                        # NEW
  ↓
  training.training(
      batch_size=args.batch_size,      # Auto-configured values
      ...
  )
  ↓
  [rest unchanged]
```

## Files to Modify (Minimal)
1. **main.py**: Add args, call auto-config, save config (~50 lines)
2. **net.py**: Add depths parameter to SwinUNETR (~10 lines)
3. **training.py**: Accept network_depth param (~5 lines)

## Files to Create (New)
1. **hardware/detection.py** (~150 lines)
2. **hardware/memory_model.py** (~300 lines)
3. **hardware/validator.py** (~100 lines)
4. **auto_config/configurator.py** (~200 lines)
5. **auto_config/heuristics.py** (~150 lines)
6. **auto_config/config_manager.py** (~100 lines)

Total new code: ~1000 lines (isolated modules)
Total modifications: ~65 lines (minimal impact)
