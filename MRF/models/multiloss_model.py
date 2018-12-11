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



class MultilossModel(SimpleModel):
    def name(self):
        return 'Multiloss Model'

    def initialize(self, opt):
        SimpleModel.initialize(self, opt)

    def test(self):
        self.real_A = self.input_A
        self.ground_B = self.input_B
        self.var_mask = self.input_mask
        self.netG_A.eval()
        self.real_A_input = self.real_A[:,:,:(self.real_A.size(2)//16)*16,:(self.real_A.size(3)//16)*16]
        with torch.no_grad():
            out = self.netG_A(self.real_A_input)

        self.pixloss_step = {}
        self.fake_B_multi = {}
        for i, fake_B in enumerate(out):
            fake_B_temp = fake_B
            fake_B = torch.zeros_like(self.ground_B)
            fake_B[:,:,:(self.real_A.size(2)//16)*16,:(self.real_A.size(3)//16)*16] = fake_B_temp
            self.fake_B_multi['step'+str(i)] = fake_B
            self.pixloss_step['step'+str(i)] = self.backloss_comp(fake_B, self.ground_B, self.var_mask)

        self.fake_B = fake_B
        dif = ((self.fake_B-self.ground_B)/self.ground_B).abs()
        self.relerr = (dif*self.var_mask).sum()/self.var_mask.sum()

        self.netG_A.train()

    def backward_G(self):
        if self.var_mask.data.sum() == 0:
            print('000000000000000000000000')
            return

        self.pixloss_step = {}
        self.fake_B_multi = {}
        out = self.netG_A(self.real_A)
        for i, fake_B in enumerate(out):
            self.pixloss_step['step'+str(i)] = self.backloss_comp(fake_B, self.ground_B, self.var_mask)
            self.fake_B_multi['step'+str(i)] = fake_B

        self.fake_B = fake_B
        self.dif = ((self.fake_B-self.ground_B)/self.ground_B).abs()
        self.relerr = (self.dif*self.var_mask).sum()/self.var_mask.sum()

        self.pixloss = self.pixloss_step['step0'] * 0
        for k, v in self.pixloss_step.items():
            self.pixloss += v

        self.pixloss.backward()

    def get_current_errors(self):
        err = OrderedDict(self.pixloss_step)
        err['relerr'] = self.relerr
        for k,v in err.items():
            err[k] = v.item()
        return err

    def get_current_visuals(self):
        fake_B_multi = {}
        for k,v in self.fake_B_multi.items():
            fake_B = self.fake_B_multi[k]
            fake_B = fake_B.cpu().float().numpy().transpose(3,2,1,0)[:,:,:,0]
            fake_B = util.inverse_preprocess_tissue_property(fake_B)
            fake_B_multi[k] = fake_B

        ground_B = self.ground_B.cpu().float().numpy().transpose(3,2,1,0)[:,:,:,0]
        mask = self.var_mask.cpu().float().numpy().transpose(3,2,1,0)[:,:,:,0]
        image_path = self.image_paths[0]
        ground_B = util.inverse_preprocess_tissue_property(ground_B)

        visuals = OrderedDict(fake_B_multi)
        visuals['ground_B'] = ground_B
        visuals['mask'] = mask
        visuals['image_path'] = image_path

        return visuals
