"""Copy input NIfTIs to a RAM-backed tmpfs directory before training.

The public entry point is setup_ram_cache(). Everything else is internal.
"""
import logging
import os
import shutil
from pathlib import Path
from typing import List

from tqdm import tqdm


def is_tmpfs(path: Path) -> bool:
    """Return True if path lives on a tmpfs mount."""
    path = Path(path).resolve()
    best_match = None
    try:
        with open('/proc/mounts', 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mount_point = Path(parts[1])
                fstype = parts[2]
                try:
                    if path == mount_point or mount_point in path.parents:
                        if best_match is None or len(str(mount_point)) > len(str(best_match[0])):
                            best_match = (mount_point, fstype)
                except ValueError:
                    continue
    except OSError:
        return False
    return best_match is not None and best_match[1] == 'tmpfs'


def _collect_paths(split_lists: list) -> list:
    """Return a flat list of all unique file paths from normalised split_lists."""
    seen = set()
    paths = []
    for fold in split_lists:
        for entry in fold:
            for v in entry.values():
                if isinstance(v, str) and v not in seen:
                    seen.add(v)
                    paths.append(Path(v))
    return paths


def _compute_path_map(src_paths: list, dest_root: Path) -> dict:
    """Compute {src: dest} mapping without touching the filesystem."""
    if not src_paths:
        return {}
    try:
        common = Path(os.path.commonpath([str(p) for p in src_paths]))
    except ValueError:
        common = Path('/')
    if common.is_file():
        common = common.parent
    return {src: dest_root / (src.relative_to(common) if src != common else Path(src.name))
            for src in src_paths}


def _copy_files(path_map: dict) -> None:
    """Copy files in path_map where dest is missing or has a different size."""
    to_copy = [
        (src, dest) for src, dest in path_map.items()
        if src.stat().st_size != (dest.stat().st_size if dest.exists() else -1)
    ]

    if to_copy:
        logging.info(f'RAM cache: copying {len(to_copy)}/{len(path_map)} files')
        for src, dest in tqdm(to_copy, desc='Copying to RAM', unit='file'):
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
    else:
        logging.info(f'RAM cache: all {len(path_map)} files already present, skipping copy')


def _remap_split_lists(split_lists: list, path_map: dict) -> list:
    """Return a new split_lists with all file path values replaced via path_map."""
    str_map = {str(k): str(v) for k, v in path_map.items()}
    remapped = []
    for fold in split_lists:
        remapped_fold = []
        for entry in fold:
            remapped_fold.append({k: str_map.get(v, v) for k, v in entry.items()})
        remapped.append(remapped_fold)
    return remapped


def setup_ram_cache(split_lists: list, dest_root: Path) -> list:
    """Copy all NIfTIs referenced by split_lists to dest_root and return remapped split_lists.

    Raises ValueError if dest_root is not on a tmpfs mount or has insufficient space.
    Files already present with matching sizes are skipped (idempotent).
    The input split_lists is not mutated.
    """
    dest_root = Path(dest_root)

    if not is_tmpfs(dest_root):
        raise ValueError(
            f'--ram-dir target {dest_root} is not on a tmpfs mount. '
            f'Check with: findmnt {dest_root}'
        )

    src_paths = _collect_paths(split_lists)
    if not src_paths:
        logging.warning('RAM cache: split_lists contains no file paths, nothing to copy.')
        return split_lists

    dest_root.mkdir(parents=True, exist_ok=True)
    path_map = _compute_path_map(src_paths, dest_root)

    # Only count bytes for files that actually need copying — files already present
    # with matching sizes are already occupying space in tmpfs and need no extra room.
    bytes_to_copy = sum(
        src.stat().st_size for src, dest in path_map.items()
        if src.exists() and src.stat().st_size != (dest.stat().st_size if dest.exists() else -1)
    )
    free_bytes = shutil.disk_usage(dest_root).free
    if bytes_to_copy > free_bytes:
        raise ValueError(
            f'RAM cache: insufficient space in {dest_root}. '
            f'Need to copy {bytes_to_copy / 1024**3:.1f} GB, '
            f'only {free_bytes / 1024**3:.1f} GB free.'
        )

    _copy_files(path_map)
    return _remap_split_lists(split_lists, path_map)
