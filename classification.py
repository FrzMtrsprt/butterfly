import codecs
import json
import os
from typing import Tuple

import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import AlexNet

from annotation_parser import parse_xml


def classify(image_path: str, rect: Tuple[int, int, int, int]) -> Tuple[int, float]:

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    model = AlexNet(num_classes=94)
    model.load_state_dict(torch.load('./weights/AlexNet.pth'))
    model.eval()

    model = model.to(device)

    image = cv2.imread(image_path)

    image_rect = image[rect[1]:rect[3], rect[0]:rect[2]]
    image_rect = transform(image_rect).to(device)
    image_rect = image_rect.unsqueeze(0)

    # classify the image
    with torch.no_grad():
        output = model(image_rect)
        prediction = torch.max(output, dim=1)[1]
        probability = torch.max(output, dim=1)[0]

    return prediction, probability


if __name__ == "__main__":
    
    for files in os.listdir('./datasets/JPEGImages'):

        image_path = os.path.join('./datasets/JPEGImages', files)
        anno_path = os.path.join('./datasets/Annotations', files.replace('.jpg', '.xml'))

        with codecs.open('categories.json', 'r', 'utf-8') as f:
            dx_to_class = json.load(f)

        annotation = parse_xml(anno_path)
        for obj in annotation['objects']:

            box = obj['bndbox']
            rect = (box['xmin'], box['ymin'], box['xmax'], box['ymax'])
            prediction, probability = classify(image_path, rect)

            prediction = dx_to_class[str(int(prediction))]

            print(f'Prediction: {prediction}, '
                f'Probability: {float(probability)}, '
                f'Label: {obj["name"]}')
