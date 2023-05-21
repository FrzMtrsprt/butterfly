import os
from typing import List, Tuple

import cv2
import pandas as pd

from annotation_parser import BndBox, parse_xml

# speed-up using multithreads
cv2.setUseOptimized(True)
cpu_count = os.cpu_count()
cv2.setNumThreads(cpu_count if cpu_count else 1)

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()


def selective_search(image_path: str, strategy_quality: bool = False) -> List[BndBox]:

    if not os.path.exists(image_path):
        raise Exception(f'image path does not exist: {image_path}')
    im = cv2.imread(image_path)

    # resize image
    new_h = 200
    scale_rate = new_h / im.shape[0]
    new_y = int(im.shape[1]*scale_rate)
    im = cv2.resize(im, (new_y, new_h))

    ss.setBaseImage(im)
    if strategy_quality:
        ss.switchToSelectiveSearchQuality()
    else:
        ss.switchToSelectiveSearchFast()

    rects = ss.process()

    bndboxes: List[BndBox] = []
    for rect in rects:
        scaled_rect = (int(rect[0] / scale_rate),  # left
                       int(rect[1] / scale_rate),  # top
                       int(rect[2] / scale_rate),  # width
                       int(rect[3] / scale_rate))  # height
        bndbox = BndBox(xmin=scaled_rect[0], ymin=scaled_rect[1],
                        xmax=scaled_rect[0] + scaled_rect[2],
                        ymax=scaled_rect[1] + scaled_rect[3])
        bndboxes.append(bndbox)

    return bndboxes


def IoU(box1: BndBox, box2: BndBox) -> float:

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


if __name__ == '__main__':

    image_dir = 'datasets/JPEGImages'
    anno_dir = 'datasets/Annotations'
    prop_dir = 'datasets/Proposals'
    if not os.path.exists(prop_dir):
        os.mkdir(prop_dir)

    positive_num = 0
    negative_num = 0

    for i, image_file in enumerate(os.listdir(image_dir)):

        image_path = os.path.join(image_dir, image_file)
        anno_path = os.path.join(anno_dir, image_file.replace('jpg', 'xml'))

        proposals = selective_search(image_path, True)
        annotation = parse_xml(anno_path)
        objects = [obj['bndbox'] for obj in annotation['objects']]

        # Divide proposals into positive and negative samples
        positive_list: List[Tuple[BndBox, float]] = []
        negative_list: List[Tuple[BndBox, float]] = []
        for prop in proposals:
            iou = 0
            for obj in objects:
                iou = IoU(prop, obj)
            if iou > 0.5:
                positive_list.append((prop, iou))
            else:
                negative_list.append((prop, iou))

        # Increase threshhold until there are no more than 10 positives
        # And take 5 of them
        threshhold = 0.5
        while len(positive_list) > 10:
            threshhold += 0.1
            positive_list = [item for item in positive_list if item[1] > threshhold]
        positive_list = positive_list[:5]

        # Balance positive and negative samples
        negative_list = [item for item in negative_list if item[1] == 0.0]
        negative_list = negative_list[:len(positive_list)]

        # Save proposals as csv files
        frame = pd.DataFrame({'xmin': [prop[0]['xmin'] for prop in positive_list + negative_list],
                              'ymin': [prop[0]['ymin'] for prop in positive_list + negative_list],
                              'xmax': [prop[0]['xmax'] for prop in positive_list + negative_list],
                              'ymax': [prop[0]['ymax'] for prop in positive_list + negative_list],
                              'label': [1 for _ in positive_list] + [0 for _ in negative_list],
                              'iou': [prop[1] for prop in positive_list + negative_list]})
        frame.to_csv(os.path.join(
            prop_dir, image_file.replace('jpg', 'csv')), index=False)

        positive_num += len(positive_list)
        negative_num += len(negative_list)

        # Print progress every 10 images
        if i % 10 == 0:
            print(f'{i+1}/{len(os.listdir(image_dir))}: {image_file}')
