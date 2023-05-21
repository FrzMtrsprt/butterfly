import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from annotation_parser import BndBox, parse_xml


class ClassificationDataset(Dataset[Tensor]):
    """
    Creates a dataset of cropped object images for classification.
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


class DetectionDataset(Dataset[Tensor]):
    """
    Creates a dataset to evaluate the proposed bounding boxes.
    """

    def __init__(self, image_dir: str,
                 prop_dir: str,
                 transform: Optional[Callable[..., Tensor]]) -> None:

        self.transform = transform
        self.item_list: List[Tuple[str, Any]] = []

        for image_file in os.listdir(image_dir):

            image_path = os.path.join(image_dir, image_file)
            prop_path = os.path.join(
                prop_dir, image_file.replace('jpg', 'csv'))

            frame = pd.read_csv(prop_path)

            # Crop images according to the proposals
            for i in range(len(frame)):
                prop = frame.iloc[i]
                self.item_list.append((image_path, prop))

    def __len__(self) -> int:

        return len(self.item_list)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        
        image_path = self.item_list[index][0]
        prop = self.item_list[index][1]

        with open(image_path, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
        
        image = image.crop((prop['xmin'], prop['ymin'],
                            prop['xmax'], prop['ymax']))
        label = int(prop['label'])

        if self.transform:
            image = self.transform(image)

        return image, label
