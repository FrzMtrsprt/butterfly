import time
from typing import List, Tuple

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import AlexNet

from selective_search import selective_search

import util


def detect(image_path: str, nms: float) -> List[Tuple[Tuple[int, int, int, int], float]]:

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    model = AlexNet(num_classes=2)
    model.load_state_dict(torch.load('./weights/Detection.pth'))
    model.eval()

    model = model.to(device)

    image = cv2.imread(image_path)

    rects = [(box['xmin'], box['ymin'], box['xmax'], box['ymax'])
             for box in selective_search(image_path)]

    # A list of tuples, each tuple is ((xmin, ymin, xmax, ymax), probability)
    positive_list: List[Tuple[Tuple[int, int, int, int], float]] = []

    for rect in rects:
        # Crop the imageand transform it
        image_rect = image[rect[1]:rect[3], rect[0]:rect[2]]
        image_rect = transform(image_rect).to(device)
        image_rect = image_rect.unsqueeze(0)

        # If probability is greater than 0.9, add it to the list
        with torch.no_grad():
            output = model(image_rect)
            output = nn.functional.softmax(output, dim=1)
            probability = output[0][1].item()
            if probability > 0.9:
                positive_list.append((rect, probability))

    nms_list = util.nms(positive_list, nms)

    return nms_list


if __name__ == "__main__":

    image_path = './datasets/TestData/IMG_000001.jpg'
    image = cv2.imread(image_path)

    start = time.time()
    nms_list = detect(image_path, 0)
    end = time.time()
    print('Detection finished in {:.3f}s'.format(end - start))

    for rect, score in nms_list:
        cv2.rectangle(image,
                      pt1=(rect[0], rect[1]),
                      pt2=(rect[2], rect[3]),
                      color=(0, 0, 255),
                      thickness=2)
        cv2.putText(image,
                    text="{:.3f}".format(score),
                    org=(rect[0], rect[1]),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 0, 255),
                    thickness=2)

    cv2.imshow('image', cv2.resize(image, (1600, 900)))
    cv2.waitKey(0)
