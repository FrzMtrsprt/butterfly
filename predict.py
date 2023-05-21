import codecs
import json
import os
import time
from typing import Dict, List, Tuple

from classification import classify
from detection import detect

if __name__ == '__main__':

    image_dir = 'datasets/TestData'
    image_list = os.listdir(image_dir)

    with codecs.open('categories.json', 'r',  'utf-8') as f:
        index_to_label = json.load(f)

    result: Dict[str, List[Tuple[str, float, int, int, int, int]]] = {}
    for dx in index_to_label.keys():
        result[dx] = []

    for i, image_name in enumerate(image_list):
        print(image_name)
        image_path = os.path.join(image_dir, image_name)

        start = time.time()
        detection = detect(image_path, 0)
        end = time.time()
        # print(f'Detection time: {end - start}')

        for rect, _ in detection:

            start = time.time()
            id, score = classify(image_path, rect)
            end = time.time()
            # print(f'Classification time: {end - start}')

            label = index_to_label[str(id)]
            # print(label, score)

            result[str(id)].append((os.path.splitext(image_name)[0], score,
                                    int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])))

        # Save result every 10 images
        if i % 10 == 0:
            json.dump(result, open('result.json', 'w'))
