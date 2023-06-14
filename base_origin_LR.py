#59줄부터 136줄이 원본이랑 바뀐 코드고, 현재는 Left모델 학습으로 작성됨. Right모델 학습시에는 106,107줄 코드 바꾸기
#250줄 모델명 바꿔주기
# python native
import os
import json
import random
import datetime
from functools import partial


# external library
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold
import albumentations as A

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models

# visualization
import matplotlib.pyplot as plt
import wandb
import random
# 데이터 경로를 입력하세요

IMAGE_ROOT = "/opt/ml/input/data/train-LR/DCM"
LABEL_ROOT = "/opt/ml/input/data/train-LR/outputs_json"

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}
BATCH_SIZE = 4
LR = 1e-4
RANDOM_SEED = 21

NUM_EPOCHS = 77    # CHANGE
VAL_EVERY = 1

SAVED_DIR = "./workspace/results_baseline/"

if not os.path.isdir(SAVED_DIR):                                                           
    os.mkdir(SAVED_DIR)
    

pngsR = {
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
    for root, _dirs, files in os.walk(IMAGE_ROOT)
    for fname in files
    if fname.endswith("_R.png")
}
pngsL = {
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
    for root, _dirs, files in os.walk(IMAGE_ROOT)
    for fname in files
    if fname.endswith("_L.png")
}

jsonsR = {
    os.path.relpath(os.path.join(root, fname), start=LABEL_ROOT)
    for root, _dirs, files in os.walk(LABEL_ROOT)
    for fname in files
    if fname.endswith("_R.json")
}
jsonsL = {
    os.path.relpath(os.path.join(root, fname), start=LABEL_ROOT)
    for root, _dirs, files in os.walk(LABEL_ROOT)
    for fname in files
    if fname.endswith("_L.json")
}
# check the size of the dataset
# jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}
# pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}

# # assert len(jsons_fn_prefix - pngs_fn_prefix) == 0
# jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}
# pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}

# assert len(jsons_fn_prefix - pngs_fn_prefix) == 0
# assert len(pngs_fn_prefix - jsons_fn_prefix) == 0

pngsR = sorted(pngsR)
pngsL=sorted(pngsL)
jsonsR=sorted(jsonsR)
jsonsL = sorted(jsonsL)
# define dataset class

from sklearn.model_selection import KFold

class XRayDataset(Dataset):
    def __init__(self, is_train=True, transforms=None):
        #left 모델 학습중. Right로 하고 싶으면 pngsR, jsonsR로 바꿔주면됨
        _filenames = np.array(pngsL)
        _labelnames = np.array(jsonsL)
        
        # dummy label
        ys = [0 for fname in _filenames]
        
        # 전체 데이터의 20%를 validation data로 쓰기 위해 `n_splits`를
        # 5으로 설정하여 KFold를 수행합니다.
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        filenames = []
        labelnames = []
        for i, (train_idx, val_idx) in enumerate(kf.split(_filenames)):
            if is_train:
                # 0번을 validation dataset으로 사용합니다.
                if i == 0:
                    continue
                filenames += list(_filenames[train_idx])
                labelnames += list(_labelnames[train_idx])
            
            else:
                filenames = list(_filenames[val_idx])
                labelnames = list(_labelnames[val_idx])
                
                # skip i > 0
                break
        
        self.filenames = filenames
        self.labelnames = labelnames
        self.is_train = is_train
        self.transforms = transforms
    

    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(IMAGE_ROOT, image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        label_name = self.labelnames[item]
        label_path = os.path.join(LABEL_ROOT, label_name)
        
        # process a label of shape (H, W, NC)
        label_shape = tuple(image.shape[:2]) + (len(CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)
        
        # read label file
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]
        
        # iterate each class
        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])
            
            # polygon to mask
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label
        
        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transforms(**inputs)
            
            image = result["image"]
            label = result["mask"] if self.is_train else label

        # to tenser will be done later
        image = image.transpose(2, 0, 1)    # make channel first
        label = label.transpose(2, 0, 1)
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
            
        return image, label
# check data sample
# define colors
PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]

# utility function
# this does not care overlap
def label2rgb(label):
    image_size = label.shape[1:] + (3, )
    image = np.zeros(image_size, dtype=np.uint8)
    
    for i, class_label in enumerate(label):
        image[class_label == 1] = PALETTE[i]
        
    return image

tf = A.Resize(512, 512)
train_dataset = XRayDataset(is_train=True, transforms=tf)
valid_dataset = XRayDataset(is_train=False, transforms=tf)
image, label = train_dataset[0]

print(image.shape, label.shape) # 3channels  and 29 class\
    
    
# setup dataloader

train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=BATCH_SIZE,
    shuffle=True,
    #num_workers=8,
    num_workers=4,
    drop_last=True,
)

valid_loader = DataLoader(
    dataset=valid_dataset, 
    batch_size=1, # why different?
    shuffle=False,
    num_workers=1, # why different?
    drop_last=False
)

# define functions for training
def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    
    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

# 폴더 이름 정하기.
import time
timestr = time.strftime("%m-%d-%H:%M")

saving_file_name = f'fcn_resnet50_best_model_R.pt'
def save_model(model, file_name=saving_file_name):
    output_path = os.path.join(SAVED_DIR, file_name)
    torch.save(model, output_path)
    

def set_seed():
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    

def validation(epoch, model, data_loader, criterion, thr=0.5):
    print(f'Start validation #{epoch:2d}')
    model.eval()

    dices = []
    with torch.no_grad():
        n_class = len(CLASSES)
        total_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, masks = images.cuda(), masks.cuda()         
            model = model.cuda()
            
            outputs = model(images)['out']
            
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            
            # restore original size
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
            
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu()
            masks = masks.detach().cpu()
            
            dice = dice_coef(outputs, masks)
            dices.append(dice)
                
    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(CLASSES, dices_per_class)
    ]
    dice_str = "\n".join(dice_str)
    print(dice_str)
    
    avg_dice = torch.mean(dices_per_class).item()
    
    wandb.log({"avg_dice": avg_dice})
    for i, c in enumerate(CLASSES):
        wandb.log({f"dice_{c}": dices_per_class[i].item()})
    return avg_dice

wandb.init(
    # set the wandb project where this run will be logged
    project="Segmentation",
    entity="hi-ai",
    name=f"baseline_{timestr}",
    # track hyperparameters and run metadata
    config={
    "learning_rate": LR,
    "architecture": "pretrained",
    "dataset": "hand_bone_image",
    "epochs": NUM_EPOCHS,
    "seed": RANDOM_SEED,
    "batch_size":BATCH_SIZE
    }
)


def train(model, data_loader, val_loader, criterion, optimizer):
    print(f'Start training..')
    
    n_class = len(CLASSES)
    best_dice = 0.
    
    for epoch in range(NUM_EPOCHS):
        model.train()

        for step, (images, masks) in enumerate(data_loader):            
            # gpu 연산을 위해 device 할당
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()
            
            # inference
            outputs = model(images)['out']
            
            # loss 계산
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # step 주기에 따른 loss 출력
            if (step + 1) % 10 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )
                wandb.log({"Epoch": epoch+1,"Loss":round(loss.item(),4)}) # vallidation 계산할때마다 로스 계산.
            
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % VAL_EVERY == 0:
            dice = validation(epoch + 1, model, val_loader, criterion)
            
            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {SAVED_DIR}")
                best_dice = dice
                save_model(model)
    #save_model(model)                
        
        
# training
model = models.segmentation.fcn_resnet50(pretrained=True)
# model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
# deeplab
# deeplabv3_resnet50 deeplabv3_resnet101

# FCN
# LRASPP
# lraspp_mobilenet_v3_large

# output class를 data set에 맞도록 수정
model.classifier[4] = nn.Conv2d(512, len(CLASSES), kernel_size=1) # classifier[4] 는 무슨 뜻이지?

# Loss function 정의
criterion = nn.BCEWithLogitsLoss()

# Optimizer 정의
optimizer = optim.Adam(params=model.parameters(), lr=LR, weight_decay=1e-6)

set_seed()

train(model, train_loader, valid_loader, criterion, optimizer)
model = torch.load(os.path.join(SAVED_DIR, saving_file_name))

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
# 데이터 경로 설정
data_dir = '/opt/ml/input/data/train-RL/DCM'

# 이미지 파일 로드
image_files = []
labels = []
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith('_R.png') or file.endswith('_L.png'):
            image_files.append(os.path.join(root, file))
            labels.append(file.endswith('_L.png'))

# 데이터 전처리
X = []
y = []
for image_file, label in zip(image_files, labels):
    image = cv2.imread(image_file)
    image = cv2.resize(image, (224, 224))  # 이미지 크기 조정
    image = image.astype("float32") / 255.0  # 이미지 스케일링
    X.append(image)
    y.append(label)

X = np.array(X)
y = np.array(y)

# 클래스 레이블 인코딩
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 학습 및 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 생성
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 모델 학습
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# 모델 평가
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

model.save('./workspace/results_baseline/my_model.h5')