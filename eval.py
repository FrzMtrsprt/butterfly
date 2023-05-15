import codecs
import json
import os

import torch
from torch.utils.data import DataLoader
from torchvision.models.alexnet import AlexNet
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from dataset_models import ClassificationDataset

if __name__ == '__main__':
    # load the model and evaluate it
    model_path = './weights/Alexnet.pth'
    model = AlexNet(num_classes=94)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 指定数据集目录
    image_path = os.path.abspath('datasets/JPEGImages/')
    if not os.path.exists(image_path):
        raise Exception(f"{image_path} path does not exist.")

    anno_path = os.path.abspath('datasets/Annotations/')
    if not os.path.exists(anno_path):
        raise Exception(f"{anno_path} path does not exist.")

    data_transform = Compose([Resize((224, 224)),
                              ToTensor(),
                              Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    with codecs.open('categories.json', 'r', 'utf-8') as f:
        idx_to_class = json.load(f)
    class_to_idx = {v: int(k) for k, v in idx_to_class.items()}
    
    dataset = ClassificationDataset(anno_path, image_path, class_to_idx, data_transform)
    validate_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=os.cpu_count())

    val_num = len(dataset)
    acc_sum = 0.0
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            outputs = model(val_images)
            predict_y = torch.max(outputs, dim=1)[1]
            acc_sum += torch.eq(predict_y, val_labels).sum().item()
            print(predict_y, val_labels)

    val_accurate = acc_sum / val_num
    print(f'val_accuracy: {val_accurate:.3f}')
