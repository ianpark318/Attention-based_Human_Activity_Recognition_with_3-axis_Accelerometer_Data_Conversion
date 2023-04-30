import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import pandas as pd
from dataset import RPDataset
from sklearn.preprocessing import LabelEncoder
import random
import albumentations as A
from sklearn.model_selection import train_test_split
from model import EfficientNet, ViT, ResNet, AttResNet, VGG
import warnings
import os
warnings.filterwarnings('ignore')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CFG = {
    'IMG_SIZE':224,
    'EPOCHS':30,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':32,
    'SEED':42
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED'])
def train_func(model, optimizer, scheduler, device):
    model.to(device)
    #     criterion = FocalLoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    best_val_score = 0
    best_model = None

    for epoch in range(1, CFG['EPOCHS'] + 1):
        model.train()
        train_loss = []

        val_loss = []
        preds, trues = [], []

        for i, data in enumerate(tqdm(RP_train_loader)):
            images, labels = data

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        model.eval()

        with torch.no_grad():
            for i, data in enumerate(tqdm(RP_val_loader)):
                images, labels = data

                images = images.to(device)
                labels = labels.to(device)

                logit = model(images)
                loss = criterion(logit, labels)

                val_loss.append(loss.item())

                preds += logit.argmax(1).detach().cpu().numpy().tolist()
                trues += labels.detach().cpu().numpy().tolist()

            _val_loss = np.mean(val_loss)

        _val_score = f1_score(trues, preds, average='micro')

        _train_loss = np.mean(train_loss)
        print(
            f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val F1 : [{_val_score:.5f}]')

        if scheduler is not None:
            scheduler.step(_val_score)

        if best_val_score < _val_score:
            best_val_score = _val_score
            torch.save(model, 'save_model/0429_eff_224_mfcc.pth')


train_df = pd.read_csv('data/mfcc_train_data.csv', index_col = 0)

RP_tfms = A.Compose([
    A.Resize(width=CFG['IMG_SIZE'], height=CFG['IMG_SIZE']),
    A.Normalize()
], p=1)

le = LabelEncoder()
le = le.fit(train_df['action'])
train_df['action'] = le.transform(train_df['action'])

train, val, _, _ = train_test_split(train_df, train_df['action'], test_size=0.1, random_state=CFG['SEED'], stratify=train_df['action'])
train['img_path'] = train['img_path'].apply(lambda x : x.replace('./ETRI_data_RP_png', '../ETRIdata'))
val['img_path'] = val['img_path'].apply(lambda x : x.replace('./ETRI_data_RP_png', '../ETRIdata'))


RP_train_dataset = RPDataset(df=train, rp_path_list=train['img_path'].values, label_list=train['action'].values, tfms=RP_tfms)
RP_train_loader = DataLoader(RP_train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

RP_val_dataset = RPDataset(df=val, rp_path_list=val['img_path'].values, label_list=val['action'].values, tfms=RP_tfms)
RP_val_loader = DataLoader(RP_val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

_model = EfficientNet(4)
_model = _model.to(device)
_model.eval()
_optimizer = torch.optim.Adam(params=_model.parameters(), lr=CFG["LEARNING_RATE"])
_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(_optimizer, mode='max', factor=0.5, patience=2, threshold_mode='abs', min_lr=1e-8, verbose=True)

train_func(_model, _optimizer, _scheduler, device)