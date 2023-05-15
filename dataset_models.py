import os
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Compose, PILToTensor

from annotation_parser import BndBox, parse_xml


class ButterflyDataset(Dataset[Tensor]):
    """
    Creates a dataset of object images cropped with bounding boxes.
    """

    def __init__(self, anno_dir: str,
                 image_dir: str,
                 class_to_idx: Dict[str, int],
                 transform: Optional[Callable[..., Tensor]]) -> None:

        self.image_dir = image_dir
        self.datas: List[Tuple[str, str, BndBox]] = []
        self.transform = transform
        self.class_to_idx = class_to_idx

        for image_file in os.listdir(image_dir):

            anno_file = image_file.replace('jpg', 'xml')
            anno_path = os.path.join(anno_dir, anno_file)
            annotation = parse_xml(anno_path)
            for obj in annotation['objects']:
                file_name = image_file
                name = obj['name']
                bndbox = obj['bndbox']
                self.datas.append((file_name, name, bndbox))

    def __len__(self) -> int:

        return len(self.datas)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:

        image_path = os.path.join(self.image_dir, self.datas[index][0])
        print(image_path)
        with open(image_path, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
        bndbox = self.datas[index][2]

        image = image.crop((bndbox['xmin'], bndbox['ymin'],
                            bndbox['xmax'], bndbox['ymax']))
        if self.transform:
            image = self.transform(image)

        label = self.class_to_idx[self.datas[index][1]]
        return image, label


class BndboxDataset(Dataset[Tensor]):
    """
    Creates a dataset cropped by the rects.
    """

    def __init__(self, image_path: str,
                 rects: List[Tuple[int, int, int, int]],
                 transform: Optional[Callable[..., Tensor]]) -> None:

        with open(image_path, 'rb') as f:
            image = Image.open(f)
            self.image = image.convert('RGB')
        self.rects = rects
        self.transform = transform

    def __len__(self) -> int:

        return len(self.rects)

    def __getitem__(self, index: int) -> Tensor:

        rect = self.rects[index]
        image = self.image.crop((rect[0], rect[1],
                                 rect[0] + rect[2], rect[1] + rect[3]))
        if self.transform:
            image = self.transform(image)

        return image
