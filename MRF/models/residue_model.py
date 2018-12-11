import numpy as np
import torch
import torch.nn as nn
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
from .RegreClass import FNN_Regression, FNN_FeatExtract
from .architecture import *


class ResidueModel(SimpleModel):
    def name(self):
        return 'Residue Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.opt = opt
        
        # self.input_A = self.Tensor()
        # self.input_B = self.Tensor()
        # self.input_mask = self.Tensor()

        # load/define networks
        self.networks = nn.ModuleDict({
            'FE_T1': FNN_FeatExtract(opt.input_nc, 64, 64, 4),
            'SQ_T1': Unet_3ds_struc(None, 64, 1),
            'SQres_T1': Unet_3ds_struc(None, 64, 1),
            'FE_T2': FNN_FeatExtract(opt.input_nc, 64, 64, 4),
            'SQ_T2': Unet_3ds_struc(None, 64, 1),
            'SQres_T2': Unet_3ds_struc(None, 64, 1),
            'FE_TiMap_T1': FNN_FeatExtract(2, 64, 64, 4),
            'FE_TiMap_T2': FNN_FeatExtract(2, 64, 64, 4)
            })
        self.networks.cuda(self.gpu_ids[0]).apply(networks.weights_init)
        
        if not self.isTrain or opt.continue_train:
            self.load_network(self.networks, opt.saved_model_path)

        if self.isTrain:
            # initialize optimizers
            self.optimizer_G = []
            for k in ['FE_T1', 'SQ_T1', 'SQres_T1', 'FE_T2', 'SQ_T2', 'SQres_T2']:
                self.optimizer_G.append(torch.optim.Adam(self.networks[k].parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)))
            self.optimizer_D = []
            for k in ['FE_TiMap_T1', 'FE_TiMap_T2']:
                self.optimizer_D.append(torch.optim.Adam(self.networks[k].parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)))

            self.schedulers = []
            for optimizer in self.optimizer_G + self.optimizer_D:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        # define loss functions
        self.criterion = opt.criterion
        self.backloss_comp = self.get_backloss_new(opt.criterion)
        self.feat_sim_loss = self.get_backloss_new('L2ae')

        self.outputs = {}
        self.loss = {}

        print('---------- Networks initialized -------------')
        networks.print_network(self.networks)
        print('-----------------------------------------------')
        with open(opt.file_name, 'at') as log_file:
            old_stdout = sys.stdout
            sys.stdout = log_file
            print('---------- Networks initialized -------------')
            networks.print_network(self.networks)
            print('-----------------------------------------------')
            sys.stdout = old_stdout

    def test(self):
        self.networks.eval()
        with torch.no_grad():
            self.forward()
        self.fake_B = self.outputs['fake_B']
        dif = ((self.fake_B-self.ground_B)/self.ground_B).abs()
        self.relerr = (dif*self.var_mask).sum()/self.var_mask.sum()
        self.getloss_G()
        self.getloss_D()
        self.networks.train()

    def getloss_G(self):
        if self.var_mask.data.sum() == 0:
            print('000000000000000000000000')
            return

        out = self.outputs
        gt = self.ground_B
        mask = self.var_mask
        self.loss['step0'] = self.backloss_comp(out['m'], gt, mask)
        self.loss['step1'] = self.backloss_comp(out['fake_B'], gt, mask)
        self.loss_G = self.loss['step0'] + self.loss['step1']

        self.fake_B = self.outputs['fake_B']
        dif = ((self.fake_B-self.ground_B)/self.ground_B).abs()
        self.relerr = (dif*self.var_mask).sum()/self.var_mask.sum()

    def getloss_D(self):
        fgt1 = self.networks['FE_TiMap_T1'](self.ground_B)
        fgt2 = self.networks['FE_TiMap_T2'](self.ground_B)
        fG = torch.cat([fgt1, fgt2], 1)

        m = self.outputs['m'].data.requires_grad_()
        fE = torch.cat([self.networks['FE_TiMap_T1'](m), self.networks['FE_TiMap_T2'](m)], 1)

        fM = self.outputs['f'].data.requires_grad_()

        mask = self.var_mask

        self.loss['feat_sim_GM'] = self.feat_sim_loss(fM, fG, mask)
        self.loss['feat_sim_EG'] = self.feat_sim_loss(fE, fG, mask)
        self.loss['feat_sim_EM'] = self.feat_sim_loss(fE, fM, mask)

        if self.loss['feat_sim_EM'] > 1:
            self.loss_D = self.loss['feat_sim_GM']
        else:
            self.loss_D = self.loss['feat_sim_GM'] - 0.2 * (self.loss['feat_sim_EG'] + self.loss['feat_sim_EM'])
        

    def get_current_errors(self):
        err = OrderedDict(self.loss)
        err['relerr'] = self.relerr
        for k,v in err.items():
            err[k] = v.item()
        return err

    def get_current_visuals(self):
        visuals = {}
        for k,v in self.outputs.items():
            visuals[k] = self.outputs[k].cpu().float().detach().numpy().transpose(3,2,1,0)[:,:,:,0]
        visuals['ground_B'] = self.ground_B.cpu().float().detach().numpy().transpose(3,2,1,0)[:,:,:,0]
        visuals['mask'] = self.var_mask.cpu().float().detach().numpy().transpose(3,2,1,0)[:,:,:,0]
        visuals['image_path'] = self.image_paths[0]
        
        visuals['m'] = util.inverse_preprocess_tissue_property(visuals['m'])
        visuals['fake_B'] = util.inverse_preprocess_tissue_property(visuals['fake_B'])
        visuals['ground_B'] = util.inverse_preprocess_tissue_property(visuals['ground_B'])

        visuals = OrderedDict(visuals)
        return visuals

    def forward(self):
        self.real_A = self.input_A.requires_grad_()
        self.ground_B = self.input_B.requires_grad_()
        self.var_mask = self.input_mask.requires_grad_()

        f1 = self.networks['FE_T1'](self.real_A)
        f2 = self.networks['FE_T2'](self.real_A)
        self.outputs['f'] = torch.cat([f1, f2], 1)
        m1 = self.networks['SQ_T1'](f1)
        m2 = self.networks['SQ_T2'](f2)
        self.outputs['m'] = torch.cat([m1, m2], 1)
        fm1 = self.networks['FE_TiMap_T1'](self.outputs['m'])
        fm2 = self.networks['FE_TiMap_T2'](self.outputs['m'])
        self.outputs['fm'] = torch.cat([fm1, fm2], 1)
        mres1 = m1 + 0.1 * self.networks['SQres_T1'](f1 - fm1)
        mres2 = m2 + 0.1 * self.networks['SQres_T2'](f2 - fm2)
        self.outputs['fake_B'] = torch.cat([mres1, mres2], 1)

        

    def optimize_parameters(self):
        # forward
        self.forward()

        # G_A
        for optimizer in self.optimizer_G:
            optimizer.zero_grad()
        self.getloss_G()
        self.loss_G.backward()
        for optimizer in self.optimizer_G:
            optimizer.step()

        # D
        for optimizer in self.optimizer_D:
            optimizer.zero_grad()
        self.getloss_D()
        self.loss_D.backward()
        for optimizer in self.optimizer_D:
            optimizer.step()

    def save(self, label):
        self.save_network(self.networks, 'networks', label, self.gpu_ids)

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizer_G[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

