import numpy as np
import torch
from dropout_integrated import *
import torch.utils.data

import os, sys, glob, shutil, json
import cv2

from PIL import Image
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms


class SVHNDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        # 原始SVHN中类别10为数字0
        lbl = np.array(self.img_label[index], dtype=np.int)
        lbl = list(lbl) + (5 - len(lbl)) * [10]

        return img, torch.from_numpy(np.array(lbl[:5]))

    def __len__(self):
        return len(self.img_path)

test_path = glob.glob('./test/*.png')
test_path.sort()
test_json = open('./test.json')
test_json = json.load(test_json)
test_label = [test_json[x]['label'] for x in test_json]

test_loader = torch.utils.data.DataLoader(
    SVHNDataset(test_path,test_label)
)


def predict(test_loader,model,tta = 10):
    model.eva()
    test_pred_tta=None

    for _ in range(tta):
        test_pred = []
        with torch.no_grad():
            for i ,input,target in enumerate(test_loader) :
                c1,c2,c3,c4,c5,c6 = model(test_loader[0])
                output = np.concatenate([
                    c1.data.numpy(),
                    c2.data.numpy(),
                    c3.data.numpy(),
                    c4.data.numpy(),
                    c5.data.numpy(),
                    c6.data.numpy()
                ],axis=1)
                test_pred.append(output)

                if test_pred_tta is None:
                    test_pred_tta = test_pred
                else:
                    test_pred_tta +=test_pred

            return test_pred_tta

predict = predict(test_loader,model,tta=10)