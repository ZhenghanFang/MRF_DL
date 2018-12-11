import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
# from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import sys
from .simple_model import SimpleModel
from .loss import BCEWithLogitsLoss_mask



class RegreClassModel(SimpleModel):
    def name(self):
        return 'Regresion + Classification Model'

    def initialize(self, opt):
        SimpleModel.initialize(self, opt)
        self.class_loss = BCEWithLogitsLoss_mask(pos_weight=self.Tensor([100]))
        self.thT1 = 2000/5000
        self.thT2 = 250/500

    def test(self):
        self.real_A = self.input_A
        self.ground_B = self.input_B
        self.var_mask = self.input_mask

        self.netG_A.eval()
        with torch.no_grad():
            out = self.netG_A(self.real_A)
        self.netG_A.train()

        self.fake_B_class = out['class']
        self.fake_B_class0 = out['class0']
        self.fake_B_class1 = out['class1']
        self.compute_pixloss()

        self.fake_B_T1 = (self.fake_B_class0[:, 0:1] * (self.fake_B_class[:, 0:1]<0).to(self.fake_B_class)
            + self.fake_B_class1[:, 0:1] * (self.fake_B_class[:, 0:1]>=0).to(self.fake_B_class))
        self.fake_B_T2 = (self.fake_B_class0[:, 1:2] * (self.fake_B_class[:, 1:2]<0).to(self.fake_B_class)
            + self.fake_B_class1[:, 0:1] * (self.fake_B_class[:, 1:2]>=0).to(self.fake_B_class))
        self.fake_B = torch.cat([self.fake_B_T1, self.fake_B_T2], 1)
        dif = ((self.fake_B-self.ground_B)/self.ground_B).abs()
        self.relerr = (dif*self.var_mask).sum()/self.var_mask.sum()


    def backward_G(self):
        if self.var_mask.data.sum() == 0:
            print('000000000000000000000000')
            return

        mask_T1_class1 = (self.ground_B[:, 0:1]>self.thT1).to(self.ground_B)
        mask_T1_class0 = 1 - mask_T1_class1
        mask_T2_class1 = (self.ground_B[:, 1:2]>self.thT2).to(self.ground_B)
        mask_T2_class0 = 1 - mask_T2_class1

        out = self.netG_A(self.real_A)
        self.fake_B_class = out['class']
        self.fake_B_class0 = out['class0']
        self.fake_B_class1 = out['class1']
        self.pixloss = self.compute_pixloss()
        self.pixloss.backward()

        self.relerr = self.Tensor([0])

    def get_current_errors(self):
        return OrderedDict([
            ('pixloss_class', self.pixloss_class.item()),
            ('pixloss_class0', self.pixloss_class0.item()),
            ('pixloss_class1', self.pixloss_class1.item()),
            ('relerr', self.relerr.item())
            ])

    def get_current_visuals(self):
        fake_B = self.fake_B.cpu().float().numpy().transpose(3,2,1,0)[:,:,:,0]
        ground_B = self.ground_B.cpu().float().numpy().transpose(3,2,1,0)[:,:,:,0]
        mask = self.var_mask.cpu().float().numpy().transpose(3,2,1,0)[:,:,:,0]
        image_path = self.image_paths[0]
        fake_B_class = self.fake_B_class.cpu().float().numpy().transpose(3,2,1,0)[:,:,:,0]
        fake_B = util.inverse_preprocess_tissue_property(fake_B)
        ground_B = util.inverse_preprocess_tissue_property(ground_B)

        return OrderedDict([
            ('fake_B', fake_B),
            ('ground_B', ground_B),
            ('mask', mask),
            ('image_path', image_path),
            ('fake_B_class', fake_B_class)
            ])

    def compute_pixloss(self):
        mask_T1_class1 = (self.ground_B[:, 0:1]>self.thT1).to(self.ground_B)
        mask_T1_class0 = 1 - mask_T1_class1
        mask_T2_class1 = (self.ground_B[:, 1:2]>self.thT2).to(self.ground_B)
        mask_T2_class0 = 1 - mask_T2_class1
        self.pixloss_class0 = (
            self.backloss_comp(
                self.fake_B_class0[:, 0:1], 
                self.ground_B[:, 0:1], 
                self.var_mask * mask_T1_class0) 
            + self.backloss_comp(
                self.fake_B_class0[:, 1:2], 
                self.ground_B[:, 1:2], 
                self.var_mask * mask_T2_class0)
            )
        self.pixloss_class1 = (
            self.backloss_comp(
                self.fake_B_class1[:, 0:1], 
                self.ground_B[:, 0:1], 
                self.var_mask * mask_T1_class1) 
            + self.backloss_comp(
                self.fake_B_class1[:, 1:2], 
                self.ground_B[:, 1:2], 
                self.var_mask * mask_T2_class1)
            )
        self.pixloss_class = (
            self.class_loss(self.fake_B_class[:, 0:1], mask_T1_class1, self.var_mask) 
            + self.class_loss(self.fake_B_class[:, 1:2], mask_T2_class1, self.var_mask)
            )
        return self.pixloss_class + self.pixloss_class0 + self.pixloss_class1
