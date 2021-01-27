from pathlib import Path

from bcblib.tools.nifti_utils import is_nifti


def create_input_path_list_from_root(root_folder_path, recursive_search=False):
    root_folder_path = Path(root_folder_path)
    if not root_folder_path.is_dir():
        raise ValueError(root_folder_path + ' does not exist or is not a directory')
    input_path_list = []
    if not recursive_search:
        input_path_list = [p for p in root_folder_path.iterdir() if is_nifti(p)]
    if recursive_search or len(input_path_list) == 0:
        input_path_list = [p for p in root_folder_path.rglob('*') if is_nifti(p)]
    return input_path_list
