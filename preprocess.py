import codecs
import json
import os
from typing import Dict

from PIL import Image

from annotation_parser import parse_xml


def crop_objects(image_path: str, anno_path: str, output_dir: str) -> None:
    image = Image.open(image_path)
    annotation = parse_xml(anno_path)
    for i in range(len(annotation['objects'])):
        obj = annotation['objects'][i]
        obj_name = obj['name']
        obj_image = image.crop(
            (obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']))

        category_dir = os.path.join(output_dir, obj_name)
        if not os.path.exists(category_dir):
            os.makedirs(category_dir, exist_ok=True)
        obj_image.save(os.path.join(
            category_dir, os.path.split(image_path)[1] + f'_{i}.png'), 'PNG')


def get_categories(anno_path: str) -> Dict[int, str]:
    categories: dict[int, str] = {}
    index = 0
    for file in os.listdir(anno_path):
        if file.endswith('.xml'):
            annotation = parse_xml(os.path.join('datasets/Annotations', file))
            for obj in annotation['objects']:
                if obj['name'] not in categories.values():
                    categories.update({index: obj['name']})
                    index += 1
    return categories


if __name__ == '__main__':
    # Crop objects from images
    for image_name in os.listdir('datasets/JPEGImages'):
        image_path = os.path.join('datasets/JPEGImages', image_name)
        anno_path = os.path.join('datasets/Annotations', image_name.replace(
            '.jpg', '.xml'))
        crop_objects(image_path, anno_path, 'datasets/Cropped')

    # Dump categories to json file
    categories = get_categories('datasets/Annotations')
    with codecs.open('categories.json', 'w', encoding='utf-8') as f:
        json.dump(categories, f, indent=4, ensure_ascii=False)
