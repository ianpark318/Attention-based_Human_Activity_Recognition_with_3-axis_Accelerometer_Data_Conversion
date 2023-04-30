import torch
import pandas as pd
from dataset import RPDataset, GpsDataset, MFCCDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import albumentations as A
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CFG = {
    'IMG_SIZE':224,
    'EPOCHS':30,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':32,
    'SEED':42
}

test_df = pd.read_csv('data/mfcc_test_data.csv', index_col = 0)

le = LabelEncoder()
le = le.fit(test_df['action'])
test_df['action'] = le.transform(test_df['action'])

# test_df['img_path'] = test_df['img_path'].apply(lambda x : x.replace('./ETRI_data_RP_png', '../ETRIdata'))
#

MFCC_tfms = A.Compose([
    A.Resize(width=CFG['IMG_SIZE'], height=CFG['IMG_SIZE']),
    A.Normalize()
], p=1)

# RP_tfms = A.Compose([
#     A.Resize(width=CFG['IMG_SIZE'], height=CFG['IMG_SIZE']),
#     A.Normalize()
# ], p=1)


MFCC_test_dataset = MFCCDataset(df=test_df, mfcc_path_list=test_df['img_path'].values, label_list=test_df['action'].values, tfms=MFCC_tfms)
MFCC_test_loader = DataLoader(MFCC_test_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)


def inference(model, device):
    model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(MFCC_test_loader)):
            images, labels = data

            images = images.to(device)

            logit = model(images)
            preds += logit.argmax(1).detach().cpu().numpy().tolist()
    return preds

model = torch.load('save_model/0430_res_mfcc.pth')
preds = inference(model, device)
confusion_matrix = confusion_matrix(test_df['action'], preds, labels=[x for x in range(4)])
plt.figure(figsize=(5,5))
plt.title('Confusion Matrix')

sns.heatmap(confusion_matrix, annot=True)

f1 = f1_score(test_df['action'], preds, average='micro')
print('F1-score: {0:.4f}'.format(f1))
y_true = test_df['action']
y_pred = preds
target_names = [str(x) for x in range(4)]
print(classification_report(y_true, y_pred, target_names=target_names, digits=4))
