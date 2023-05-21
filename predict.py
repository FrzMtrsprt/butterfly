import codecs
import json
import os
import time

from classification import classify
from detection import detect

if __name__ == '__main__':

    image_dir = 'datasets/TestData'
    image_list = os.listdir(image_dir)

    with codecs.open('categories.json', 'r',  'utf-8') as f:
        dx_to_class = json.load(f)

    for image_name in image_list:
        print(image_name)
        image_path = os.path.join(image_dir, image_name)

        start = time.time()
        detection = detect(image_path, 0)
        end = time.time()
        print(f'Detection time: {end - start}')

        for rect, _ in detection:

            start = time.time()
            id, score = classify(image_path, rect)
            end = time.time()
            print(f'Classification time: {end - start}')

            label = dx_to_class[str(id)]
            print(label, score)
