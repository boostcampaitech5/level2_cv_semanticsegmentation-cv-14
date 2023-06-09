import os
import albumentations as A
from schedule import set_seed
from dataset import XRayDataset, get_pngs, get_jsons, XRayInferenceDataset
from train import train
from inference import inference, get_csv
import argparse
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import wandb
import time

timestr = time.strftime("%m-%d-%H:%M")

parser = argparse.ArgumentParser()
parser.add_argument('--IMAGE_ROOT', type=str, default='/opt/ml/input/train/DCM')
parser.add_argument('--LABEL_ROOT', type=str, default='/opt/ml/input/train/outputs_json')
parser.add_argument('--SAVE_FILE_NAME', type=str, default='fcn_resnet50_best_model2.pt')
parser.add_argument('--TEST_IMAGE_ROOT', type=str, default='/opt/ml/input/test/DCM')
parser.add_argument('--BATCH_SIZE', type=int, default=8)
parser.add_argument('--LR', type=float, default=1e-4)
parser.add_argument('--NUM_EPOCHS', type=int, default=80)
parser.add_argument('--VAL_EVERY', type=int, default=1)
parser.add_argument('--RANDOM_SEED', type=int, default=21)
parser.add_argument('--SAVED_DIR', type=str, default='./workspace/results_baseline2/')
parser.add_argument('--VAL_NUM', type=int, default=0)
parser.add_argument('--logging_wandb', type=bool, default=False)
parser.add_argument('--START_TRAIN', type=bool, default=False)
parser.add_argument('--START_TEST', type=bool, default=False)
parser.add_argument('-c', '--config', help='Path to YAML configuration file')

args = parser.parse_args()

with open(args.config, 'r') as file :
     config=yaml.safe_load(file)
parser.set_defaults(**config)
args = parser.parse_args()
print('v : ', args)

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]


if args.logging_wandb :
    wandb.init(
        # set the wandb project where this run will be logged
        project="TEST_SGH_Segmentation",
        entity="hi-ai",
        name=f"test_baseline_{timestr}",
        # track hyperparameters and run metadata
        config={
        "learning_rate": args.LR,
        "architecture": "pretrained",
        "dataset": "hand_bone_image",
        "epochs": args.NUM_EPOCHS,
        "seed": args.RANDOM_SEED,
        "batch_size":args.BATCH_SIZE
        }
    )

tf = A.Resize(512, 512)

if not os.path.isdir(args.SAVED_DIR):                                                           
        os.mkdir(args.SAVED_DIR)

if args.START_TRAIN :
    pngs = get_pngs(args.IMAGE_ROOT)
    jsons = get_jsons(args.LABEL_ROOT)

    train_dataset = XRayDataset(args=args, is_train=True, transforms=tf, CLASSES=CLASSES ,pngs=pngs, jsons=jsons)
    valid_dataset = XRayDataset(args=args, is_train=False, transforms=tf, CLASSES=CLASSES,  pngs=pngs, jsons=jsons)

    train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=args.BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    drop_last=True,
    )

    valid_loader = DataLoader(
    dataset=valid_dataset, 
    batch_size=2,
    shuffle=False,
    num_workers=2,
    drop_last=False
    )

    model = models.segmentation.fcn_resnet50(pretrained=True)
    # output class를 data set에 맞도록 수정
    model.classifier[4] = nn.Conv2d(512, len(CLASSES), kernel_size=1)
    # Loss function 정의
    criterion = nn.BCEWithLogitsLoss()
    # Optimizer 정의
    optimizer = optim.Adam(params=model.parameters(), lr=args.LR, weight_decay=1e-6)

    set_seed(args.RANDOM_SEED)

    train(model, train_loader, valid_loader, criterion, optimizer, CLASSES=CLASSES, args=args)

if args.START_TEST :
    print(f'Start testing {args.SAVED_DIR +args.SAVE_FILE_NAME}')
    test_model = torch.load(os.path.join(args.SAVED_DIR, args.SAVE_FILE_NAME))
    # test_model = torch.load('/opt/ml/input/code/workspace/results_baseline/fcn_resnet50_best_model.pt')
    test_pngs = get_pngs(args.TEST_IMAGE_ROOT)
    test_dataset = XRayInferenceDataset(transforms=tf, IMAGE_ROOT=args.TEST_IMAGE_ROOT ,test_pngs=test_pngs)

    test_loader = DataLoader(
    dataset=test_dataset, 
    batch_size=2,
    shuffle=False,
    num_workers=2,
    drop_last=False
    )

    rles, filename_and_class = inference(model=test_model,CLASSES=CLASSES ,data_loader=test_loader)
    get_csv(rles, filename_and_class, args.SAVED_DIR)


