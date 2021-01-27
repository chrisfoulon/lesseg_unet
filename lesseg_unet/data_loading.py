from pathlib import Path
from collections.abc import Sequence
from typing import Tuple


def match_img_seg_by_names(img_path_list: Sequence, seg_path_list: Sequence) -> Tuple[Sequence, Sequence]:
    img_dict = {img: [les for les in seg_path_list if Path(img).name in Path(les).name][0] for img in img_path_list}
    images = list(img_dict.keys())
    segs = list(img_dict.values())
    print('Number of images: {}'.format(len(images)))
    return images, segs
