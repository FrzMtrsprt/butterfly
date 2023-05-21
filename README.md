# Butterfly Detection
This is a two-stage object detection model for detecting butterflies in images.
## Usage
### 1. Install dependencies
```
pip install -r requirements.txt
```
### 2. Prepare the dataset
Place the dataset in the following structure:
```
datasets
├─ Annotations
│  ├─ IMG_000001.xml
│  └─ ...
├─ JPEGImages
│  ├─ IMG_000001.jpg
│  └─ ...
└─ TestData
   ├─ IMG_000001.jpg
   └─ ...
```

### 3. Train the detection model
Run [train_detection.ipynb](train_detection.ipynb).  
You can test the model with [detection.py](detection.py).
### 4. Train the classification model
Run [train.ipynb](train.ipynb).  
You can test the model with [classification.py](classification.py).
### 5. Generate predictions
Run [predict.py](predict.py).
