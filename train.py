# python native
import json
import os
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts , CyclicLR, ExponentialLR,StepLR, CosineAnnealingLR

# model
import segmentation_models_pytorch as smp
# from model import FCN8s,FCN16s,FCN32s,DeepLabV3p

from loss import Focal_loss,IOU_loss,Dice_loss,Calc_loss,Dice_Focal_loss
from scheduler import CosineAnnealingWarmUpRestarts
# visualization
import matplotlib.pyplot as plt
import wandb
import random
# 데이터 경로를 입력하세요


import time
timestr = time.strftime("%m-%d-%H:%M")
## 이름 정하기1!!!
# saving_file_name = f'D3_FCN_mit_{timestr}.pt'
# TODO: 이름 정하기!!!
saving_file_name = f'MAnet_effi_b7_aug_fdloss_lossdown_{timestr}.pt'
IMAGE_ROOT = "/opt/ml/input/data/train/DCM"
LABEL_ROOT = "/opt/ml/input/data/train/outputs_json"

BATCH_SIZE = 2
LR = 7e-4
RANDOM_SEED = 21

NUM_EPOCHS = 77   # CHANGE
VAL_EVERY = 3

wandb.init(
    # set the wandb project where this run will be logged
    project="segmentation new",
    # project="Segmentation loss comparison",
    entity="hi-ai",
    name=f"{saving_file_name[:-3]}",
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


SAVED_DIR = "./workspace/results_baseline_v3/"

if not os.path.isdir(SAVED_DIR):                                                           
    os.mkdir(SAVED_DIR)

def set_seed():
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
set_seed()  

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
# check the size of the dataset
pngs = {
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
    for root, _dirs, files in os.walk(IMAGE_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
}

jsons = {
    os.path.relpath(os.path.join(root, fname), start=LABEL_ROOT)
    for root, _dirs, files in os.walk(LABEL_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".json"
}

jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}
pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}

# assert len(jsons_fn_prefix - pngs_fn_prefix) == 0
jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}
pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}

assert len(jsons_fn_prefix - pngs_fn_prefix) == 0
assert len(pngs_fn_prefix - jsons_fn_prefix) == 0

pngs = sorted(pngs)
jsons = sorted(jsons)

# define dataset class

class XRayDataset(Dataset):
    def __init__(self, is_train=True, transforms=None):
        _filenames = np.array(pngs)
        _labelnames = np.array(jsons)
        
        # split train-valid
        # 한 폴더 안에 한 인물의 양손에 대한 `.dcm` 파일이 존재하기 때문에
        # 폴더 이름을 그룹으로 해서 GroupKFold를 수행합니다.
        # 동일 인물의 손이 train, valid에 따로 들어가는 것을 방지합니다.
        groups = [os.path.dirname(fname) for fname in _filenames]
        
        # dummy label
        ys = [0 for fname in _filenames]
        
        # 전체 데이터의 20%를 validation data로 쓰기 위해 `n_splits`를
        # 5으로 설정하여 KFold를 수행합니다.
        gkf = GroupKFold(n_splits=5)
        
        filenames = []
        labelnames = []
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if is_train:
                # 0번을 validation dataset으로 사용합니다.
                if i == 0:
                    continue
                    
                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])
            
            else:
                filenames = list(_filenames[y])
                labelnames = list(_labelnames[y])
                
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
    



# # Define your data augmentation transformations
# tf = transforms.Compose([
#     transforms.RandomApply([transforms.RandomHorizontalFlip(p=0.5)]),
#     # transforms.RandomApply([transforms.RandomAdjustContrast(),], p=0.5),
#     # transforms.RandomApply([transforms.RandomElasticDeformation(),], p=0.5),
#     transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),], p=0.5),   
#     transforms.Resize(size=(512, 512)),
#     transforms.ToTensor(),
# ])

# Define your data augmentation transformations
transforms_aug = A.Compose([
    A.Resize(height=1024, width=1024),
    # A.ElasticTransform(),
    # A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    # A.ColorJitter(brightness=(1), contrast=(1), saturation=(0.8), hue=(0.5)) #always_apply=False, p=1.0, 
    # A.FancyPCA(always_apply=False, p=1.0, alpha=.74)
    A.RandomContrast(always_apply=False, p=1.0, limit=(-0.32, 0.4)),
    A.HorizontalFlip(p=0.5),
    # A.RandomContrast(always_apply=False, p=1.0, limit=(0.3, 0.4))
    
    # A.ElasticTransform(always_apply=False, p=1.0, alpha=1.0, sigma=50.0, alpha_affine=50.0, interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None, approximate=False, same_dxdy=False)
])

tf = A.Compose([
    A.Resize(height=1024, width=1024),
    # A.RandomContrast(always_apply=False, p=1.0, limit=(0.3, 0.4))
])
# Apply data augmentation to your train and validation datasets
train_dataset = XRayDataset(is_train=True, transforms=transforms_aug)
valid_dataset = XRayDataset(is_train=False, transforms=tf)

# train_dataset = XRayDataset(is_train=True, transforms=tf)
# valid_dataset = XRayDataset(is_train=False, transforms=tf)
image, label = train_dataset[0]

print(image.shape, label.shape) # 3channels  and 29 class\

    
# setup dataloader

train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=6,
    drop_last=True,
    persistent_workers=True
)

valid_loader = DataLoader(
    dataset=valid_dataset, 
    batch_size=1, 
    shuffle=False,
    num_workers=2,
    drop_last=False,
    persistent_workers=True
)

# define functions for training

def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2) # (batch, class, h*w)
    y_pred_f = y_pred.flatten(2)

    multi_pred_true = y_true_f * y_pred_f # (batch, class, h*w)
    intersection = torch.sum(multi_pred_true, -1) # (batch, class) # h*w에 대해서 다 더함.
    eps = 0.0001

    # 참인데 맞추지 못한것.
    y_true_f = y_true_f.to(torch.int)
    b = (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

    false_negative =  torch.sum((y_true_f - multi_pred_true),-1) /b
    # 거짓인데 맞춘것.
    false_positive = torch.sum(y_pred_f - multi_pred_true,-1) /b

    
    return (2. * intersection + eps) / b, false_negative, false_positive


# 폴더 이름 정하기.
def save_model(model, file_name=saving_file_name, epoch_index=0):
    saving_file_name = f'{epoch_index}_{file_name}'
    output_path = os.path.join(SAVED_DIR, saving_file_name)
    torch.save(model, output_path)
    



def validation(epoch, model, data_loader, criterion, thr=0.5):
    print(f'Start validation #{epoch:2d}')
    model.eval()

    dices = []
    fns = []
    fps = []
    with torch.no_grad():
        n_class = len(CLASSES)
        total_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, masks = images.cuda(), masks.cuda()         
            model = model.cuda()
            
            # outputs = model(images)['out']
            outputs = model(images)
            
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            
            # restore original size
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bicubic",align_corners=True) #align_corners=True 격자를 맞춰주는 작업.
            
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu()
            # outputs = outputs.detach().cpu()
            masks = masks.detach().cpu()
            
            dice,fn,fp = dice_coef(outputs, masks)
            dices.append(dice)
            fns.append(fn)
            fps.append(fp)
        # dataloader batch 별로 틀린게 있는지 확인
    print('dices per dataloader batch')
    for i in range(len(dices)):
        valid_idx_tp = dices[i]
        valid_idx_fn = fns[i] 
        valid_idx_fp = fps[i]
        
        print('dataloader batch index : ',i)
        print(f'====> TP : {torch.mean(valid_idx_tp)} FN :{torch.mean(valid_idx_fn)} FP : {torch.mean(valid_idx_fp)}')
        print(f'{i}th batch: tp: {valid_idx_tp}')
        print(f'{i}th batch: fn: {valid_idx_fn}')
        print(f'{i}th batch: fp: {valid_idx_fp}')

    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(CLASSES, dices_per_class)
    ]
    dice_str = "\n".join(dice_str)
    print(dice_str)
    fns_per_class = torch.mean(torch.cat(fns, 0), 0)
    fps_per_class = torch.mean(torch.cat(fps, 0), 0)
    avg_dice = torch.mean(dices_per_class).item()
    avg_fns = torch.mean(fns_per_class).item()
    avg_fps = torch.mean(fps_per_class).item()
    
    wandb.log({"avg_dice": avg_dice, "avg_fns": avg_fns, "avg_fps": avg_fps})
    # for i, c in enumerate(CLASSES):
    #     wandb.log({f"dice_{c}": dices_per_class[i].item()})
    return avg_dice



def train(model, data_loader, val_loader, criterion, optimizer):
    print(f'Start training..')
    
    n_class = len(CLASSES)
    best_dice = 0.

    paitence = 3
    count=0
    
    for epoch in range(NUM_EPOCHS):
        model.train()

        for step, (images, masks) in enumerate(data_loader):            
            # gpu 연산을 위해 device 할당
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()
            
            # inference
            # outputs = model(images)['out']
            outputs = model(images)
            
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
                    f'Loss: {round(loss.item(),4)}, '
                    f'LR: {round(optimizer.param_groups[0]["lr"],9)}'
                )
                wandb.log({"learning_rate":optimizer.param_groups[0]['lr'],"Epoch": epoch+1,"Loss":round(loss.item(),4)}) # vallidation 계산할때마다 로스 계산.
                # scheduler.step()
        # lr update
        scheduler.step()
        
        # 데이터 분포를 알기위햇 저장해야됨.
        if epoch==0:
            save_model(model,epoch_index=epoch)
            
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch+1) % VAL_EVERY == 0:
            dice = validation(epoch + 1, model, val_loader, criterion)
            
            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {SAVED_DIR}")
                save_model(model,epoch_index=epoch)    
                best_dice = dice
                count=0
                
            else:
                count+=1
                print('this is not best dice! count :',count)
                if count >=paitence:
                    # 얼리스타핑 - 모델 학습 종료!
                    print('#################################')
                    print('Early stopping!!!')
                    print('#################################')
                    break
    # save_model(model)


# gpu vram 32gb 거의 꽉참.
# model = smp.UnetPlusPlus(
#     encoder_name="efficientnet-b7",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     # encoder_depth =5,
#     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=29,                      # model output channels (number of classes in your dataset)
# )

model = smp.MAnet(
    encoder_name="efficientnet-b7",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    # encoder_depth =5,
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    decoder_use_batchnorm=True,
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=29,                      # model output channels (number of classes in your dataset)
)



# optimizer = optim.AdamW(params=model.parameters(), lr=LR, weight_decay=1e-6)

lr_decay_step = 1
# scheduler = StepLR(optimizer, lr_decay_step, gamma=0.95)
# scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=3e-6, last_epoch=-1)
# scheduler = CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001, step_size_up=5, step_size_down=10, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.95, last_epoch=-1, verbose=False)
# scheduler = CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001, step_size_up=5, step_size_down=10, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=- 1, verbose=False)
# scheduler = ExponentialLR(optimizer, gamma=0.95)


optimizer = optim.AdamW(model.parameters(), lr = 1e-5)
scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=30, eta_max=7e-3, T_up=7, gamma=0.7)
# scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=30, eta_max=3e-4, T_up=7, gamma=0.7)

# Loss function 정의
# criterion = nn.BCEWithLogitsLoss()

# criterion = Focal_loss()
criterion = Dice_Focal_loss()

train(model, train_loader, valid_loader, criterion, optimizer)
