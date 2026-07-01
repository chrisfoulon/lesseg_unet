"""Unit tests for lesseg_unet.ram_cache module."""
import shutil
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

import pytest

from lesseg_unet.ram_cache import (
    is_tmpfs,
    _collect_paths,
    _compute_path_map,
    _copy_files,
    _remap_split_lists,
    setup_ram_cache,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROC_MOUNTS_TMPFS = (
    "sysfs /sys sysfs rw,nosuid 0 0\n"
    "tmpfs /tmp tmpfs rw,nosuid 0 0\n"
    "ext4 /home ext4 rw 0 0\n"
)

PROC_MOUNTS_NO_TMPFS = (
    "sysfs /sys sysfs rw 0 0\n"
    "ext4 / ext4 rw 0 0\n"
    "ext4 /home ext4 rw 0 0\n"
)


def _make_split_lists(paths):
    """Build a minimal split_lists from a list of (image_path, label_path) tuples."""
    return [[{'image': str(img), 'label': str(lbl)} for img, lbl in paths]]


# ---------------------------------------------------------------------------
# is_tmpfs
# ---------------------------------------------------------------------------

class TestIsTmpfs:
    def test_returns_true_for_tmpfs_mount(self):
        with patch('builtins.open', mock_open(read_data=PROC_MOUNTS_TMPFS)):
            assert is_tmpfs(Path('/tmp/lesseg')) is True

    def test_returns_false_for_non_tmpfs_mount(self):
        with patch('builtins.open', mock_open(read_data=PROC_MOUNTS_NO_TMPFS)):
            assert is_tmpfs(Path('/home/user/data')) is False

    def test_returns_false_when_path_not_in_mounts(self):
        with patch('builtins.open', mock_open(read_data=PROC_MOUNTS_NO_TMPFS)):
            assert is_tmpfs(Path('/mnt/unknown')) is False

    def test_returns_false_on_os_error(self):
        with patch('builtins.open', side_effect=OSError):
            assert is_tmpfs(Path('/tmp')) is False

    def test_picks_deepest_matching_mount(self):
        # /tmp/ram is tmpfs even though /tmp is also listed as tmpfs
        mounts = (
            "tmpfs /tmp tmpfs rw 0 0\n"
            "tmpfs /tmp/ram tmpfs rw 0 0\n"
            "ext4 /home ext4 rw 0 0\n"
        )
        with patch('builtins.open', mock_open(read_data=mounts)):
            assert is_tmpfs(Path('/tmp/ram/data')) is True


# ---------------------------------------------------------------------------
# _collect_paths
# ---------------------------------------------------------------------------

class TestCollectPaths:
    def test_single_modal(self):
        sl = [[{'image': '/a/img.nii.gz', 'label': '/a/lbl.nii.gz'}]]
        paths = _collect_paths(sl)
        assert set(str(p) for p in paths) == {'/a/img.nii.gz', '/a/lbl.nii.gz'}

    def test_multimodal(self):
        sl = [[{'image_a': '/a/dwi.nii.gz', 'image_b': '/a/adc.nii.gz', 'label': '/a/lbl.nii.gz'}]]
        paths = _collect_paths(sl)
        assert len(paths) == 3

    def test_deduplicates_across_folds(self):
        sl = [
            [{'image': '/a/img1.nii.gz', 'label': '/a/lbl1.nii.gz'}],
            [{'image': '/a/img1.nii.gz', 'label': '/a/lbl2.nii.gz'}],
        ]
        paths = _collect_paths(sl)
        path_strs = [str(p) for p in paths]
        assert path_strs.count('/a/img1.nii.gz') == 1

    def test_empty_split_lists(self):
        assert _collect_paths([]) == []

    def test_multiple_folds(self):
        sl = [
            [{'image': '/a/img1.nii.gz', 'label': '/a/lbl1.nii.gz'}],
            [{'image': '/a/img2.nii.gz', 'label': '/a/lbl2.nii.gz'}],
        ]
        paths = _collect_paths(sl)
        assert len(paths) == 4


# ---------------------------------------------------------------------------
# _copy_files
# ---------------------------------------------------------------------------

class TestComputePathMap:
    def test_basic_mapping(self, tmp_path):
        src_dir = tmp_path / 'src'
        f1 = src_dir / 'subj01' / 'img.nii.gz'
        f2 = src_dir / 'subj02' / 'img.nii.gz'
        dest_dir = tmp_path / 'dest'
        path_map = _compute_path_map([f1, f2], dest_dir)
        assert path_map[f1] == dest_dir / 'subj01' / 'img.nii.gz'
        assert path_map[f2] == dest_dir / 'subj02' / 'img.nii.gz'

    def test_empty_returns_empty(self, tmp_path):
        assert _compute_path_map([], tmp_path) == {}

    def test_single_file(self, tmp_path):
        src = tmp_path / 'src' / 'img.nii.gz'
        dest_dir = tmp_path / 'dest'
        path_map = _compute_path_map([src], dest_dir)
        assert src in path_map


class TestCopyFiles:
    def test_copies_missing_file(self, tmp_path):
        src_dir = tmp_path / 'src'
        src_dir.mkdir()
        src_file = src_dir / 'sub' / 'img.nii.gz'
        src_file.parent.mkdir()
        src_file.write_bytes(b'data')
        dest_file = tmp_path / 'dest' / 'sub' / 'img.nii.gz'

        _copy_files({src_file: dest_file})

        assert dest_file.exists()
        assert dest_file.read_bytes() == b'data'

    def test_skips_file_with_matching_size(self, tmp_path):
        src_dir = tmp_path / 'src'
        src_dir.mkdir()
        src_file = src_dir / 'img.nii.gz'
        src_file.write_bytes(b'original')
        dest_file = tmp_path / 'dest' / 'img.nii.gz'
        dest_file.parent.mkdir()
        dest_file.write_bytes(b'original')
        mtime_before = dest_file.stat().st_mtime

        _copy_files({src_file: dest_file})

        assert dest_file.stat().st_mtime == mtime_before

    def test_recopy_on_size_mismatch(self, tmp_path):
        src_dir = tmp_path / 'src'
        src_dir.mkdir()
        src_file = src_dir / 'img.nii.gz'
        src_file.write_bytes(b'full content')
        dest_file = tmp_path / 'dest' / 'img.nii.gz'
        dest_file.parent.mkdir()
        dest_file.write_bytes(b'trunc')

        _copy_files({src_file: dest_file})

        assert dest_file.read_bytes() == b'full content'

    def test_preserves_relative_structure(self, tmp_path):
        src_root = tmp_path / 'src'
        (src_root / 'subj01').mkdir(parents=True)
        (src_root / 'subj02').mkdir(parents=True)
        f1 = src_root / 'subj01' / 'img.nii.gz'
        f2 = src_root / 'subj02' / 'img.nii.gz'
        f1.write_bytes(b'a')
        f2.write_bytes(b'b')
        dest_dir = tmp_path / 'dest'
        path_map = _compute_path_map([f1, f2], dest_dir)
        _copy_files(path_map)

        assert (dest_dir / 'subj01' / 'img.nii.gz').exists()
        assert (dest_dir / 'subj02' / 'img.nii.gz').exists()


# ---------------------------------------------------------------------------
# _remap_split_lists
# ---------------------------------------------------------------------------

class TestRemapSplitLists:
    def test_single_modal_remapped(self):
        sl = [[{'image': '/src/img.nii.gz', 'label': '/src/lbl.nii.gz'}]]
        path_map = {Path('/src/img.nii.gz'): Path('/dest/img.nii.gz'),
                    Path('/src/lbl.nii.gz'): Path('/dest/lbl.nii.gz')}
        result = _remap_split_lists(sl, path_map)
        assert result[0][0]['image'] == '/dest/img.nii.gz'
        assert result[0][0]['label'] == '/dest/lbl.nii.gz'

    def test_multimodal_remapped(self):
        sl = [[{'image_a': '/s/dwi.nii.gz', 'image_b': '/s/adc.nii.gz', 'label': '/s/lbl.nii.gz'}]]
        path_map = {
            Path('/s/dwi.nii.gz'): Path('/d/dwi.nii.gz'),
            Path('/s/adc.nii.gz'): Path('/d/adc.nii.gz'),
            Path('/s/lbl.nii.gz'): Path('/d/lbl.nii.gz'),
        }
        result = _remap_split_lists(sl, path_map)
        entry = result[0][0]
        assert entry['image_a'] == '/d/dwi.nii.gz'
        assert entry['image_b'] == '/d/adc.nii.gz'
        assert entry['label'] == '/d/lbl.nii.gz'

    def test_original_not_mutated(self):
        sl = [[{'image': '/src/img.nii.gz', 'label': '/src/lbl.nii.gz'}]]
        original_value = sl[0][0]['image']
        path_map = {Path('/src/img.nii.gz'): Path('/dest/img.nii.gz'),
                    Path('/src/lbl.nii.gz'): Path('/dest/lbl.nii.gz')}
        _remap_split_lists(sl, path_map)
        assert sl[0][0]['image'] == original_value

    def test_unmapped_paths_pass_through(self):
        sl = [[{'image': '/src/img.nii.gz', 'label': '/src/lbl.nii.gz'}]]
        path_map = {Path('/src/img.nii.gz'): Path('/dest/img.nii.gz')}
        result = _remap_split_lists(sl, path_map)
        # label not in path_map — should be unchanged
        assert result[0][0]['label'] == '/src/lbl.nii.gz'


# ---------------------------------------------------------------------------
# setup_ram_cache — integration
# ---------------------------------------------------------------------------

class TestSetupRamCache:
    def test_raises_if_not_tmpfs(self, tmp_path):
        with patch('builtins.open', mock_open(read_data=PROC_MOUNTS_NO_TMPFS)):
            with pytest.raises(ValueError, match='not on a tmpfs mount'):
                setup_ram_cache([[]], tmp_path)

    def test_raises_if_insufficient_space(self, tmp_path):
        with patch('lesseg_unet.ram_cache.is_tmpfs', return_value=True):
            with patch('lesseg_unet.ram_cache.shutil.disk_usage') as mock_du:
                mock_du.return_value = MagicMock(free=0)
                src = tmp_path / 'src' / 'img.nii.gz'
                src.parent.mkdir()
                src.write_bytes(b'x' * 100)
                sl = [[{'image': str(src), 'label': str(src)}]]
                dest = tmp_path / 'dest'
                with pytest.raises(ValueError, match='insufficient space'):
                    setup_ram_cache(sl, dest)

    def test_end_to_end(self, tmp_path):
        # Create fake source files
        src_dir = tmp_path / 'src'
        (src_dir / 'subj01').mkdir(parents=True)
        img = src_dir / 'subj01' / 'trace.nii.gz'
        lbl = src_dir / 'subj01' / 'lesion.nii.gz'
        img.write_bytes(b'image data')
        lbl.write_bytes(b'label data')

        dest_dir = tmp_path / 'dest'
        sl = [[{'image': str(img), 'label': str(lbl)}]]

        with patch('lesseg_unet.ram_cache.is_tmpfs', return_value=True):
            result = setup_ram_cache(sl, dest_dir)

        # Remapped paths point to dest
        assert result[0][0]['image'].startswith(str(dest_dir))
        assert result[0][0]['label'].startswith(str(dest_dir))
        # Files actually exist at new locations
        assert Path(result[0][0]['image']).exists()
        assert Path(result[0][0]['label']).exists()
        # Original split_lists unchanged
        assert sl[0][0]['image'] == str(img)

    def test_idempotent_on_second_call(self, tmp_path):
        src_dir = tmp_path / 'src'
        src_dir.mkdir()
        img = src_dir / 'img.nii.gz'
        lbl = src_dir / 'lbl.nii.gz'
        img.write_bytes(b'img')
        lbl.write_bytes(b'lbl')

        dest_dir = tmp_path / 'dest'
        sl = [[{'image': str(img), 'label': str(lbl)}]]

        with patch('lesseg_unet.ram_cache.is_tmpfs', return_value=True):
            result1 = setup_ram_cache(sl, dest_dir)
            # Record mtimes after first copy
            mtime1 = Path(result1[0][0]['image']).stat().st_mtime
            result2 = setup_ram_cache(sl, dest_dir)
            mtime2 = Path(result2[0][0]['image']).stat().st_mtime

        assert mtime1 == mtime2  # file was not re-copied

    def test_returns_original_on_empty_split_lists(self, tmp_path):
        with patch('lesseg_unet.ram_cache.is_tmpfs', return_value=True):
            result = setup_ram_cache([], tmp_path)
        assert result == []
