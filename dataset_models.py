import os
from typing import Callable, Dict, List, Optional, Tuple

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

    def __init__(self, anno_path: str,
                 image_path: str,
                 proposals: List[BndBox],
                 transform: Optional[Callable[..., Tensor]],
                 IoU: float = 0.5) -> None:

        with open(image_path, 'rb') as f:
            image = Image.open(f)
            self.image = image.convert('RGB')
        annotation = parse_xml(anno_path)
        self.objects = [obj['bndbox'] for obj in annotation['objects']]
        self.proposals = proposals
        self.transform = transform
        self.IoU = IoU

    def __len__(self) -> int:

        return len(self.proposals)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:

        prop = self.proposals[index]
        image = self.image.crop((prop['xmin'], prop['ymin'],
                                 prop['xmax'], prop['ymax']))
        
        # If IoU > 0.5, the bounding box is marked as positive.
        label = 0
        for obj in self.objects:
            if self._IoU(prop, obj) > self.IoU:
                label = 1
                break

        if self.transform:
            image = self.transform(image)

        return image, label
    
    @staticmethod
    def _IoU(box1: BndBox, box2: BndBox) -> float:

        x_left = max(box1['xmin'], box2['xmin'])
        y_top = max(box1['ymin'], box2['ymin'])
        x_right = min(box1['xmax'], box2['xmax'])
        y_bottom = min(box1['ymax'], box2['ymax'])

        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (box1['xmax'] - box1['xmin']) * (box1['ymax'] - box1['ymin'])
        box2_area = (box2['xmax'] - box2['xmin']) * (box2['ymax'] - box2['ymin'])
        union_area = box1_area + box2_area - intersection_area

        return intersection_area / union_area
