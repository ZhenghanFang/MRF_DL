import torch
import torch.nn as nn
import functools
from .block import * 
from .architecture import *

class UniNet_RegreClass_postprocess_block(nn.Module):
    def __init__(self, num_feat, norm_layer=nn.BatchNorm2d):
        super(UniNet_RegreClass_postprocess_block, self).__init__()

        self.classification = nn.Sequential(
            norm_layer(num_feat),
            nn.ReLU(True),
            conv_block(num_feat, num_feat),
            conv_block(num_feat, num_feat),
            nn.Conv2d(num_feat, 1, kernel_size=1, padding=0)
            )
        self.est_class0 = nn.Sequential(
            norm_layer(num_feat),
            nn.ReLU(True),
            conv_block(num_feat, num_feat),
            conv_block(num_feat, num_feat),
            nn.Conv2d(num_feat, 1, kernel_size=1, padding=0)
            )
        self.est_class1 = nn.Sequential(
            norm_layer(num_feat),
            nn.ReLU(True),
            conv_block(num_feat, num_feat),
            conv_block(num_feat, num_feat),
            nn.Conv2d(num_feat, 1, kernel_size=1, padding=0)
            )

    def forward(self, input):
        return {'class': self.classification(input), 
            'class0': self.est_class0(input),
            'class1': self.est_class1(input)
            }

class UniNet_RegreClass_postprocess(nn.Module):
    def __init__(self, num_feat, norm_layer=nn.BatchNorm2d):
        super(UniNet_RegreClass_postprocess, self).__init__()
        self.num_feat = num_feat
        self.T1_postprocess = UniNet_RegreClass_postprocess_block(num_feat, norm_layer)
        self.T2_postprocess = UniNet_RegreClass_postprocess_block(num_feat, norm_layer)

    def forward(self, input):
        T1 = self.T1_postprocess(input[:,0:self.num_feat])
        T2 = self.T2_postprocess(input[:,self.num_feat:2*self.num_feat])
        out = {}
        for k,v in T1.items():
            out[k] = torch.cat([T1[k], T2[k]], 1)
        return out

class UniNet_RegreClass(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UniNet_RegreClass, self).__init__()
        num_feat = opt.num_D
        self.UniNet = UniNet_init(opt, input_nc, num_feat, ngf, norm_layer, use_dropout, gpu_ids)
        self.RegreClass = UniNet_RegreClass_postprocess(num_feat, norm_layer)

    def forward(self, input):
        feat = self.UniNet(input)
        return self.RegreClass(feat)

class RegreClass(nn.Module):
    def __init__(self, input_nc):
        super(RegreClass, self).__init__()
        num_feat = 64

        self.classify_T1 = nn.Sequential(
            FNN_FeatExtract(input_nc, num_feat, num_feat=num_feat),
            Unet_3ds_struc(None, num_feat, 1)
            )
        self.classify_T2 = nn.Sequential(
            FNN_FeatExtract(input_nc, num_feat, num_feat=num_feat),
            Unet_3ds_struc(None, num_feat, 1)
            )
        self.quantify_T1_C0 = nn.Sequential(
            FNN_FeatExtract(input_nc, num_feat, num_feat=num_feat),
            Unet_3ds_struc(None, num_feat, 1)
            )
        self.quantify_T2_C0 = nn.Sequential(
            FNN_FeatExtract(input_nc, num_feat, num_feat=num_feat),
            Unet_3ds_struc(None, num_feat, 1)
            )
        self.quantify_T1_C1 = FNN_Regression(input_nc, 1)
        self.quantify_T2_C1 = FNN_Regression(input_nc, 1)

    def forward(self, input):
        label_T1 = self.classify_T1(input)
        label_T2 = self.classify_T2(input)
        map_T1_C0 = self.quantify_T1_C0(input)
        map_T2_C0 = self.quantify_T2_C0(input)
        map_T1_C1 = self.quantify_T1_C1(input)
        map_T2_C1 = self.quantify_T2_C1(input)
        return {'class': torch.cat([label_T1, label_T2], 1),
            'class0': torch.cat([map_T1_C0, map_T2_C0], 1),
            'class1': torch.cat([map_T1_C1, map_T2_C1], 1)
            }


class FNN_Regression(nn.Module):
    def __init__(self, input_nc, output_nc, num_feat=64, num_layer=4, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(FNN_Regression, self).__init__()

        ngf = [int(num_feat) for k in range(num_layer)]

        model = [conv_block(input_nc, ngf[0], kernel_size=1, padding=0, norm_layer=norm_layer)]
        for k in range(num_layer - 1):
            model.append(conv_block(ngf[k], ngf[k+1], kernel_size=1, padding=0, norm_layer=norm_layer))
        model.append(nn.Conv2d(ngf[-1], output_nc, kernel_size=1, padding=0))

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class FNN_FeatExtract(nn.Module):
    def __init__(self, input_nc, output_nc, num_feat=64, num_layer=4, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(FNN_FeatExtract, self).__init__()

        ngf = [int(num_feat) for k in range(num_layer)]

        model = [conv_block(input_nc, ngf[0], kernel_size=1, padding=0, norm_layer=norm_layer)]
        for k in range(num_layer - 1):
            model.append(conv_block(ngf[k], ngf[k+1], kernel_size=1, padding=0, norm_layer=norm_layer))
        model.append(conv_block(ngf[-1], output_nc, kernel_size=1, padding=0, norm_layer=norm_layer))

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

