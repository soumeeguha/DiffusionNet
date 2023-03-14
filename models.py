from email.mime import image
from multiprocessing import set_forkserver_preload
from turtle import forward
from typing import final
import torch
import torch.nn as nn
import torch.nn.functional as F

    

class NetworkAttentionSpChImproved(nn.Module): #------
    def __init__(self, device, features_in, img_size, num_classes, conv_ch_features, conv_ch_img, map_size, code_size = 300, attn_filter = 15, k = 30):
        super().__init__()

        self.features_in = features_in
        self.attn_filter = attn_filter
        self.conv_ch_features = conv_ch_features
        self.map_size = map_size
        self.img_size = img_size
        self.linear_features = self.map_size*self.map_size + self.img_size*self.img_size 
        self.linear1 = nn.Linear(self.features_in, self.map_size*self.map_size)
        self.relu = nn.ReLU(inplace = True)
        self.sigmoid = nn.Sigmoid()

        self.conv_img = nn.Sequential(
            nn.Conv2d(1, conv_ch_img, 3, 1, 1),
            nn.LeakyReLU(0.4),
            nn.Conv2d(conv_ch_img, conv_ch_img, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(conv_ch_img, 5, 1, 1, 0),
            nn.Sigmoid(),
        )


        self.spatial_attn = nn.Sequential(
            nn.Conv2d(5, 10, 4, 2, 0),
            nn.LeakyReLU(0.4),
            nn.ConvTranspose2d(10, 5, 4, 2, 0),
            nn.LeakyReLU(0.4),
        )

        self.channel_attn = nn.Sequential(
            nn.Conv2d(5, 1, 3, 1, 1),
            nn.LeakyReLU(0.4),
        )

        self.skip_conn = nn.Sequential(
            nn.Conv2d(6, 1, 3, 1, 1),
            nn.LeakyReLU(0.4),
        )

        self.conv_features = nn.Sequential(
            nn.Conv2d(1, conv_ch_features, 3, 1, 1),
            nn.BatchNorm2d(conv_ch_features),
            nn.ReLU(inplace = True),
            nn.Conv2d(conv_ch_features, 2*conv_ch_features, 3, 1, 1),
            nn.BatchNorm2d(2*conv_ch_features),
            nn.ReLU(inplace = True),
            nn.Conv2d(2*conv_ch_features, 4*conv_ch_features, 3, 1, 1),
            nn.BatchNorm2d(4*conv_ch_features),
            nn.ReLU(inplace = True),
            nn.Conv2d(4*conv_ch_features, 1, 1, 1, 0),
            nn.ReLU(inplace = True),
        )

        self.linear2 = nn.Linear(self.linear_features, self.linear_features)
        self.k = k
        self.linear3 = nn.Linear(self.linear_features, num_classes)
        self.softmax = nn.Softmax(dim = 1)
        self.attnwghts = (torch.ones((attn_filter, attn_filter))/(attn_filter*attn_filter)).view(1, 1, attn_filter, attn_filter).to(device)
        self.mean = (torch.ones((attn_filter, attn_filter))/(attn_filter*attn_filter)).view(1, 1, attn_filter, attn_filter).to(device)

    def forward(self, x, img):
        o1 = self.relu(self.linear1(x))
        o1 = torch.reshape(o1, (-1, 1, self.map_size, self.map_size))
        o2 = self.conv_features(o1)
        o2 = torch.flatten(o2, start_dim = 1)
        o3 = self.conv_img(img)
        spatial_attn_o3 = self.spatial_attn(o3)
        channel_attn_o3 = self.channel_attn(o3*spatial_attn_o3)
        skipped_o3 = self.skip_conn(torch.cat((channel_attn_o3, o3), 1))
        o4 = torch.cat((o2, torch.flatten(skipped_o3, start_dim = 1)), -1)
        final = self.linear3(self.relu(self.linear2(o4)))
        return skipped_o3, final
