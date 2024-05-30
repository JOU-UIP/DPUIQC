from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from transformers import Transformer
from posencode import PositionEmbeddingSine
from inceptionresnetv2 import inceptionresnetv2

class Conv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class model_qa(nn.Module):
    def __init__(self,num_classes=3,d_model=256,nheadt=8,num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.1,normalize_before=False,
                 return_attention_map=False):

        super(model_qa,self).__init__()
        # base_model = inceptionresnetv2(num_classes=1000, pretrained='imagenet')
        self.base = nn.Sequential(*list(base_model.children())[:7])

        self.stem1 = nn.Sequential(
            Conv(192, 128, (3, 3), (2, 2)),
            Conv(128, 128, (3, 3), (2, 2)),
            Conv(128, 128, (3, 3), (1, 1)),
            nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2))
        )

        self.stem2 = nn.Sequential(
            Conv(192, 128, (3, 3), (2, 2)),
            Conv(128, 128, (3, 3), (2, 2)),
            Conv(128, 128, (3, 3), (1, 1)),
            nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2))
        )

        self.transformer = Transformer(d_model=d_model, nhead=nheadt,n_patches=25,
                                       num_encoder_layers=num_encoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       normalize_before=normalize_before,
                                       dropout=dropout)


        # self.fc = nn.Linear(128, num_classes)

        self.fc = nn.Sequential(
            nn.Linear(6656, 2048),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(2048),
            nn.Dropout(p=0.25),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.25),
            nn.Linear(1024, 256),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.25),
            nn.Linear(256, num_classes)
        )
        self.return_attention_map=return_attention_map
    def forward(self, x):

        x1, x2 = torch.split(x, [512, 512], dim=2)

        x1 = self.base(x1)
        x2 = self.base(x2)

        x1 = self.stem1(x1)
        x2 = self.stem2(x2)

        x = torch.cat([x1, x2], 1)


        # sub transformer model

        x,attention_weights = self.transformer(x)
        x=self.fc(torch.flatten(x,start_dim=1))
        if self.return_attention_map:
            return x,attention_weights
        else:
            return x
class model_qa1(nn.Module):
    def __init__(self,num_classes=3,d_model=128,nheadt=8,num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.1,normalize_before=False,
                 return_attention_map=False):

        super(model_qa1,self).__init__()
        # base_model = inceptionresnetv2(num_classes=1000, pretrained='imagenet')
        base_model = inceptionresnetv2(num_classes=1000,pretrained=False)
        self.base = nn.Sequential(*list(base_model.children())[:7])

        self.stem1 = nn.Sequential(
            Conv(192, 128, (3, 3), (2, 2)),
            Conv(128, 128, (3, 3), (2, 2)),
            Conv(128, 128, (3, 3), (1, 1)),
            nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2))
        )

        self.stem2 = nn.Sequential(
            Conv(192, 128, (3, 3), (2, 2)),
            Conv(128, 128, (3, 3), (2, 2)),
            Conv(128, 128, (3, 3), (1, 1)),
            nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2))
        )

        self.transformer = Transformer(d_model=d_model, nhead=nheadt,n_patches=50,
                                       num_encoder_layers=num_encoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       normalize_before=normalize_before,
                                       dropout=dropout)


        # self.fc = nn.Linear(128, num_classes)
        # self.fc = nn.Linear(6528, num_classes)
        self.fc = nn.Sequential(
            nn.Linear(6528, 2048),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(2048),
            nn.Dropout(p=0.25),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.25),
            nn.Linear(1024, 256),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.25),
            nn.Linear(256, num_classes)
        )
        self.return_attention_map=return_attention_map
    def forward(self, x):

        x1, x2 = torch.split(x, [512, 512], dim=2)

        x1 = self.base(x1)
        x2 = self.base(x2)

        x1 = self.stem1(x1)
        x2 = self.stem2(x2)

        x = torch.cat([x1, x2], -1)

        # cnn submodel
        # x_1 = self.inception1_1(x)
        # x_2 = self.inception1_2(x)
        # x_3 = self.inception1_3(x)
        # x_4 = self.inception1_4(x)
        #
        # x = torch.cat([x_1, x_2, x_3, x_4], 1)
        # x = self.conv1(x)
        #
        # x_1 = self.reduction1_1(x)
        # x_2 = self.reduction1_2(x)
        #
        # x = torch.cat([x_1, x_2], 1)
        #
        # x_1 = self.inception2_1(x)
        # x_2 = self.inception2_2(x)
        # x_3 = self.inception2_3(x)
        # x_4 = self.inception2_4(x)
        #
        # x = torch.cat([x_1, x_2, x_3, x_4], 1)
        # x = self.conv2(x)
        #
        # x_1 = self.reduction2_1(x)
        # x_2 = self.reduction2_2(x)
        #
        # x = torch.cat([x_1, x_2], 1)
        #
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        # sub transformer model

        x,attention_weights = self.transformer(x)
        # x=self.fc(x[:,:,0])
        x=self.fc(torch.flatten(x,start_dim=1))
        if self.return_attention_map:
            return x,attention_weights
        else:
            return x


