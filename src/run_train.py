'''
@Author: Nguyen Truong Duy
@Purpose: thuc hien training
'''

import torch
import pytorch_lightning as pl
import torchvision.transforms as T
from model import BreastCancerModel
from data import load_dataloader

from lightning.pytorch import loggers as pl_loggers

# CONFIG:
INPUT_SIZE = (256,256)
ARCH = 'Unet' #     Unet, UnetPlusPlus, MAnet,Linknet,FPN,PSPNet,DeepLabV3,DeepLabV3Plus,PAN,UPerNet,Segformer,
ENCODER_NAME = 'resnet34' #mobilenet_v2 or efficientnet-b7 
EPOCH = 50
LR = 2e-4
T_MAX=50
TEST_SIZE=0.2
LR = 2e-4

SAVE_DIR = '/workspace/competitions/Sly/CV_Final_Final/experiment'

DATASET_PATH = '/workspace/competitions/Sly/CV_Final_Final/data/train'

TRANSFORM_TRAIN = T.Compose([
    T.ToTensor(),
])

TRANSFORM_EVAL = T.Compose([
    T.ToTensor(),
    #T.RandomHorizontalFlip(p=0.5),   
    #T.RandomVerticalFlip(p=0.5),   
    #T.RandomRotation(degrees=30)
])

# RUNNING

train_loader, val_loader = load_dataloader(
    dataset_path=DATASET_PATH,
    test_size=TEST_SIZE,
    input_size=INPUT_SIZE,
    transform=TRANSFORM_TRAIN
)

model = BreastCancerModel(
    arch=ARCH,
    encoder_name=ENCODER_NAME,
    in_channels=3, 
    out_classes=1,
    lr=LR,
    t_max=T_MAX
)

tb_logger = pl_loggers.TensorBoardLogger(save_dir=SAVE_DIR, name='TensorBoardLogger')
csv_logger = pl_loggers.CSVLogger(save_dir=SAVE_DIR, name='CSVLogger')

trainer = pl.Trainer(
    max_epochs=EPOCH,
    log_every_n_steps=1,
    accelerator="gpu",
    devices=[1,],
    logger=[tb_logger, csv_logger],
)


trainer.fit(
    model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
)
