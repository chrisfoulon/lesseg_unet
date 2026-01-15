# Unified Conditional Pipeline - User's Vision

## Core Concept

**NOT a master function** - Instead, a **conditional pipeline in main.py** where:
- Common stages use shared functions
- Mode-specific stages apply conditionally
- Pipeline logic is explicit and visible

## Modes and Their Requirements

### 1. Training Mode
```
Inputs:  images + labels + (optional) controls
Process: List → Match → Shuffle → Split into folds
Output:  split_lists = [[fold0_dicts], [fold1_dicts], ...]
```

### 2. Validation Mode
```
Inputs:  images + labels
Process: List → Match → (NO shuffle)
Output:  flat_list = [dict1, dict2, ...]
```

### 3. Segmentation Mode
```
Inputs:  images only (no labels)
Process: List → Match → (NO shuffle)
Output:  flat_list = [dict1, dict2, ...]
```

## Input Types (All Modes Can Use Any)

### Type A: Folder-per-modality
```
-p /data/dwi /data/adc          # Image modalities
-lp /data/lesion                # Label classes (optional)
-cp /data/controls_dwi          # Control modalities (optional, training only)
```

### Type B: Pre-matched lists
```
-li images.txt                  # Image paths
-lli labels.txt                 # Label paths
```

### Type C: Pre-split JSON (proposed --subject-list)
```
--subject-list split_lists.json # Already matched and split
```

### Type D: Segmentation dict
```
--seg-input-dict {...}          # Subfolders for segmentation
```

## Shared Stages (Common Functions)

### Stage 0→1: List Files
**Function**: `list_nifti_from_folders(folders, pattern)`
- **Used by**: ALL modes (when input is folders)
- **Input**: `{modality: folder_path}`
- **Output**: `{modality: {subject_id: file_path}}`

**Key insight**: This function is **agnostic** to whether it's listing images, labels, or controls!
```python
# Images
image_modalities = list_nifti_from_folders(
    {'dwi': '/data/dwi', 'adc': '/data/adc'},
    pattern=r'(sub-\d+)'
)

# Labels (same function!)
label_classes = list_nifti_from_folders(
    {'lesion': '/data/lesion'},
    pattern=r'(sub-\d+)'
)

# Controls (same function, different pattern!)
control_modalities = list_nifti_from_folders(
    {'dwi': '/data/controls_dwi'},
    pattern=r'(ctr-\d+)'  # Different pattern, that's all!
)
```

### Stage 1→2: Match by Subject ID
**Function**: `match_modalities_by_subject(image_modalities, label_classes, control_modalities)`
- **Used by**: ALL modes (when input is folders)
- **Input**: Stage 1 format `{modality: {subject_id: path}}`
- **Output**: `(subject_dicts, control_dicts)` where each dict has sorted keys

**Key insight**: Already handles images, labels, AND controls!
```python
subject_dicts, control_dicts = match_modalities_by_subject(
    image_modalities={'dwi': {...}, 'adc': {...}},
    label_classes={'lesion': {...}},        # None for segmentation
    control_modalities={'dwi': {...}}       # None for validation/segmentation
)
# Returns:
# subject_dicts = [
#     {'image_adc': path, 'image_dwi': path, 'label_lesion': path},
#     ...
# ]
# control_dicts = [
#     {'control_dwi': path},
#     ...
# ]
```

## Mode-Specific Stages

### Stage 2→3: Shuffle and Split (TRAINING ONLY)
**Function**: `shuffle_and_split_subjects(subject_dicts, n_folds, shuffle=True, random_seed=42)`
- **Used by**: Training mode ONLY
- **Input**: `[dict1, dict2, ...]` (flat list)
- **Output**: `[[fold0], [fold1], ...]` (nested lists)

```python
# Training: merge subjects + controls, then shuffle+split
all_subjects = subject_dicts + control_dicts
split_lists = shuffle_and_split_subjects(
    all_subjects,
    n_folds=5,
    shuffle=True,
    random_seed=42
)

# Validation/Segmentation: just use flat list
all_subjects = subject_dicts + control_dicts  # Already done!
```

## Proposed Pipeline in main.py (Pseudocode)

```python
# Detect mode
is_training = (args.checkpoint is None)
has_labels = (label_folders_dict is not None)

# ===== STAGE 0→1: List files (ALL modes, folder input only) =====
if using_folder_mode:
    # Images (required for all modes)
    image_modalities = list_nifti_from_folders(
        image_folders_dict,
        args.subject_pattern
    )

    # Labels (training + validation, optional for segmentation)
    label_classes = None
    if label_folders_dict:
        label_classes = list_nifti_from_folders(
            label_folders_dict,
            args.subject_pattern
        )

    # Controls (training only)
    control_modalities = None
    if control_folders_dict and is_training:
        control_modalities = list_nifti_from_folders(
            control_folders_dict,
            args.control_pattern  # Different pattern!
        )
###USER COMMENT: Actually here, I would have the "elif using_list_mode:" because as I said, the lists might not be matched so here we'd load the lists as it's already done in the code and feed them to the stage 1->2 matching stage
    # ===== STAGE 1→2: Match by subject ID (ALL modes) =====
    subject_dicts, control_dicts = match_modalities_by_subject(
        image_modalities=image_modalities,
        label_classes=label_classes,
        control_modalities=control_modalities,
        require_all=True
    )

    # Merge subjects and controls
    all_subjects = subject_dicts + control_dicts

    # ===== STAGE 2→3: Shuffle+split (TRAINING ONLY) =====
    if is_training:
        img_list = shuffle_and_split_subjects(
            subject_dicts=all_subjects,
            n_folds=args.folds_number,
            shuffle=True,
            random_seed=42
        )
        # Format: [[fold0], [fold1], ...]
    else:
        img_list = all_subjects
        # Format: [dict1, dict2, ...]

elif using_list_mode:
    # Different input path, skips Stage 0→1
    img_list = file_to_list(args.input_list)
    # ...
###USER COMMENT: even it's in final format, we need some sanity check to make sure the listed paths are accessible, unless it is done later in the code?
elif using_presplit_mode:
    # Skips all stages, already in final format
    with open(args.subject_list) as f:
        img_list = json.load(f)
    # ...
```

## Why This is Better Than a Master Function

### Master Function Approach (what I was doing):
```python
# Hidden complexity, unclear what happens
img_list = process_folder_mode_input(
    ...,
    shuffle=is_training  # Magic happens inside
)
```
**Problems:**
- Pipeline logic hidden in function
- Hard to see what stages are shared
- Difficult to add new input types
- Not clear which stages run for which modes

### Conditional Pipeline Approach (what you want):
```python
# Explicit stages, clear flow
image_modalities = list_nifti_from_folders(...)      # Stage 0→1
subject_dicts, _ = match_modalities_by_subject(...)  # Stage 1→2

if is_training:
    img_list = shuffle_and_split_subjects(...)       # Stage 2→3
else:
    img_list = subject_dicts                         # Stop at Stage 2
```
**Benefits:**
- ✅ Pipeline logic visible in main.py
- ✅ Clear which stages are shared
- ✅ Easy to add new input types (just add new conditional branch)
- ✅ Easy to see differences between modes
- ✅ Functions are pure utilities, not orchestrators

## What Happens to folder_mode_to_split_lists()?

**Option 1: Keep it as backward-compatible wrapper** (for existing code)
```python
def folder_mode_to_split_lists(...):
    """Legacy wrapper - calls stage functions internally."""
    # Stages 0→1→2→3 (always shuffle for backward compat)
```
###USER COMMENT: let's deprecate it but keep it until we are certain we didn't break anything
**Option 2: Deprecate it** (mark for removal after migration)
```python
@deprecated("Use stage functions directly in main.py")
def folder_mode_to_split_lists(...):
    ...
```

**Option 3: Remove it entirely** (if no external dependencies)

## Implementation Plan

### Step 1: Verify Current State
- ✅ Stage functions exist and work
- ✅ Old function uses stage functions internally
- ⬜ Identify all call sites

### Step 2: Refactor main.py
- ⬜ Replace training path with explicit stages
- ⬜ Replace inference path with explicit stages
- ⬜ Remove duplication
- ⬜ Add clear comments for each stage

### Step 3: Handle folder_mode_to_split_lists()
- ⬜ Decide: keep as wrapper, deprecate, or remove
- ⬜ Update tests if needed

### Step 4: Test Everything
- ⬜ All existing tests pass
- ⬜ Training produces same splits
- ⬜ Inference doesn't shuffle
- ⬜ Validation doesn't shuffle

## Questions for User

1. **Is this understanding correct?** You want the pipeline explicit in main.py, not hidden?

2. **What should happen to `folder_mode_to_split_lists()`?**
   - Keep as backward-compatible wrapper?
   - Deprecate with warning?
   - Remove entirely?

3. **Should I show you the refactored main.py code before implementing?**

4. **Any other input types or modes I'm missing?**

###USER COMMENTS: You are complicating things and mixing things up. 

The patterns: we do not want to force the user to have specific patterns like "sub"-something by default it is simply not realistic (just using the tool with french data would break the pattern). If the user is not providing a pattern, the default is NOTHING. The users CAN use a patternto filter the images but this is likely not scalable, most datasets are aggregates of multiple sites so I don't see how we can do the job of the user on this one. They have to provide an easy to find pattern or organise their folders explicitely otherwise we can't cover all the possibilities. BUT, a pattern that would be more useful would be a matching_pattern, so we can match the different modalities with each other. 
The modality names can be patterns too. If all the images were in the same folder, I think it's easier if I give the pattern for the modality like "adc" or "dwi" rather that the subject number. Then the images can be matches by taking the residual of the pattern, no? Do you think that could be our default (no pattern at all is the default but obiousvly that wouldn't work in a single folder for all modalities)? Like if the input is a folder for the modalities dwi adc and the label "label" we can use the modality names as pattern (if they are not provided) and search the corresponding images in the folder and then we take the paths we listed and remove the pattern ans see if the rest matches with ONE and only one image for each other modality / label. Do you think that works? And is that generic enough to work both with separate folder or single folder? Make sure you plan properly and test the different cases (simple cases, as I said we can handle everything). 
For the folders input I would handle 3 different cases: 
-single folder with matching pattern (as we pretty but NEED patterns for that case, the argument could be ONE -p folder_path but multiple modality_names [and optionally patterns]. And if there is only -p folder_path we assume there is only one modality (but we also accept -p folder_path modality_name name0 [pattern pattern0])) 
-one folder per modality either there is no matching pattern so we NEED all the images to exactly correspond between folders OR we have a matching pattern 
-BIDS datasets which should be pretty simple (except for the derivatives but we could say the the derivatives folder should have the same structure as the BIDS folder and so the patterns should be the same but just starting at the derivatives folder instead of the root). BIDS is the lowest priority right now though so could be skipped. 
Tell me if we cover a large PRACTICAL set of cases without putting too much complexity on us? 

What I imagine (tell me if I am missing something or if it's actually suboptimal): 
We'd had two sets of functions one for the input loading depending on the type and sanity check of the paths and one for the formatting. The idea is that the formatting functions are not aware that there are different input types that's the role of the loading functions to format them to the right stage. 
So we'd have a function to read ONE folder (of course any function can be devided into other functions for example, we could have a function for simple folders and one for BIDS). 
This function should return a list of image paths. 

And another function would read ONE list file (I think this function exists already) and return a list of image paths. But because we haven't listed the files ourselves, we need to check the files actually exist. We will assume the user want to use the whole list and not bother with patterns here. 

Then we have our first formatting function. It takes the different lists that we should have at this point (up to a list per modality + a list of labels + a list per control modality).
The goal is to match uniquely all the images. So here we'd use the matching_patterns if it is provided or we use the modality names OR if we have neither we assume all the filenames match between the lists. 
So we create our list of dicts for the subjects e.g. {'image_adc': path, 'image_dwi': path, 'label_lesion': path} and the one for the controls (which could require a ctr_matching_pattern in case of multiple modalities or just use the modality names if that works)

And here, we have the function that takes a file that already has matched list of dicts (either for the subjects or the controls) in which we just need to extract the paths and verify they exist. 

So at this point, either from the formatting function of the loading function we have a matched list of dicts (or two if we have controls)

Here if we have the checkpoint option, we have enough for either the validation (if we have the labels) or the segmentation. 

If not, we just need the split_lists so either we take the previously loaded or formatted list of dicts and shuffle it according to the parameters in the last formatting function or we load the split_lists file (or files if we have a controls split_lists files) in the last loading function that checks that the files exist. 

Tell me if you think it's a good strategy or not based on the existing code and if that solves our issues and particularly, if that, in the end, simplifies the data loading and formatting (and make it more flexible and scalable) or not (don't agree with me by default). 