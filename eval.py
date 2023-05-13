import os

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from alexnet import AlexNet

if __name__ == '__main__':
    # load the model and evaluate it
    model_path = './weights/Alexnet.pth'
    model = AlexNet(num_classes=94)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    data_path = os.path.abspath(os.path.join(os.getcwd(), "datasets/Cropped"))
    data_transform = Compose([Resize((224, 224)),
                              ToTensor(),
                              Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    validate_dataset = ImageFolder(root=data_path, transform=data_transform)
    validate_loader = DataLoader(
        validate_dataset, batch_size=4, shuffle=False, num_workers=os.cpu_count())
    val_num = len(validate_dataset)
    acc_sum = 0.0
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            outputs = model(val_images)
            predict_y = torch.max(outputs, dim=1)[1]
            acc_sum += torch.eq(predict_y, val_labels).sum().item()

    val_accurate = acc_sum / val_num
    print(f'val_accuracy: {val_accurate:.3f}')
