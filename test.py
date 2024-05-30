from __future__ import print_function
from __future__ import division

import json

import torch
import numpy as np
from torchvision import datasets, models, transforms
from PIL import Image
import os
import pandas as pd
from DP_UIQC import model_qa,model_qa1
import glob
from tqdm import tqdm
import shutil

print("PyTorch Version: ",torch.__version__)
#print("Torchvision Version: ",torchvision.__version__)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0")

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize([1024, 512]),
#         transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),
    'val': transforms.Compose([
         transforms.Resize([1024, 512]),
#         transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}


if __name__=="__main__":
    flag==2:
    KonCept512 = model_qa1(num_classes=3,num_encoder_layers=6)
    KonCept512.load_state_dict(torch.load('./model/newUIQC_cls.pth'))
    KonCept512.eval().to(device)
    imgs=glob.glob('/media/underwater/xzr/underwater_computer_TS/PUIQC/测试/目标检测/拼对/395/*.png')
    imgs.sort(key=lambda x: int(os.path.splitext(os.path.split(x)[1])[0].split('-')[0]))
    img_batch = torch.zeros(1, 3, 1024, 512).to(device)
    t=1
    num=0

    for k in range(0, len(imgs)):
        img_transforms = data_transforms['val'](Image.open(imgs[k]))
        img_batch[0]=img_transforms
        output = KonCept512(img_batch)
        prob=np.array(output.argmax(1).detach().cpu())[0]
        print(imgs[k],prob)