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
import time


class SimpleModel(BaseModel):
    def name(self):
        return 'Simple Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.opt = opt

        
        self.input_A = self.Tensor()
        self.input_B = self.Tensor()
        self.input_mask = self.Tensor()

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        
        self.netG_A = networks.define_G(opt, opt.input_nc, opt.output_nc,
            opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids)
        self.load_network(self.netG_A, opt.saved_model_path)
        print(self.netG_A.model_T1)

        self.netG_A = networks.define_G(opt, opt.input_nc, opt.output_nc,
            opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids)
        if self.opt.gan:
            self.netD_A = networks.define_D(input_nc=4,
                                        which_model_netD='n_layers', gpu_ids=self.gpu_ids)
            self.criterion_D = networks.GANLoss(use_lsgan=True, tensor=self.Tensor)

        
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG_A, opt.saved_model_path)

        if self.isTrain:
            # self.old_lr = opt.lr
            # define loss functions
            # self.criterion = torch.nn.MSELoss()
            # initialize optimizers
            self.optimizers = []
            # print(list(self.netG_A.parameters()))
            if opt.PCA:
                params = list(self.netG_A.parameters())
                self.optimizers.append( torch.optim.Adam((x for x in params[1:]), lr=opt.lr, betas=(opt.beta1, 0.999)) )
                self.optimizers.append( torch.optim.Adam((x for x in params[0:1]), lr=opt.lr_PCA, betas=(opt.beta1, 0.999)) )
            else:
                self.optimizers.append( torch.optim.Adam(self.netG_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)) )

            if self.opt.gan:
                self.optimizer_D = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)) 

            self.schedulers = []
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))
            if self.opt.gan:
                self.schedulers.append(networks.get_scheduler(self.optimizer_D, opt))

        self.criterion = opt.criterion
        self.get_backloss()
        self.grad = opt.gradloss
        self.gradloss = Variable(self.Tensor([0]))

        self.multiloss = opt.multiloss
        self.multiloss_f = opt.multiloss_f

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        if self.opt.gan:
            networks.print_network(self.netD_A)
        print('-----------------------------------------------')
        with open(opt.file_name, 'at') as log_file:
            old_stdout = sys.stdout
            sys.stdout = log_file
            print('---------- Networks initialized -------------')
            networks.print_network(self.netG_A)
            if self.opt.gan:
                networks.print_network(self.netD_A)
            print('-----------------------------------------------')
            sys.stdout = old_stdout



    def set_input(self, input):
        # input_A = input['A']
        # input_B = input['B']
        # input_mask = input['mask']
        # self.input_A.resize_(input_A.size()).copy_(input_A)
        # self.input_B.resize_(input_B.size()).copy_(input_B)
        # self.input_mask.resize_(input_mask.size()).copy_(input_mask)
        self.input_A = input['A'].to(self.device)
        self.input_B = input['B'].to(self.device)
        self.input_mask = input['mask'].to(self.device)
        self.image_paths = input['A_paths']
        

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.ground_B = Variable(self.input_B)
        self.var_mask = Variable(self.input_mask)

    def test(self):
        # self.real_A = Variable(self.input_A, volatile=True)
        # self.ground_B = Variable(self.input_B, volatile=True)
        # self.var_mask = Variable(self.input_mask, volatile=True)
        self.real_A = self.input_A
        self.ground_B = self.input_B
        self.var_mask = self.input_mask
        self.netG_A.eval()
        self.fake_B = self.ground_B.clone()
        self.fake_B.data.zero_()


        self.real_A_input = self.real_A[:,:,:(self.real_A.size(2)//16)*16,:(self.real_A.size(3)//16)*16]

        with torch.no_grad():
            out = self.netG_A.forward(self.real_A_input)

        self.fake_B[:,:,:(self.real_A.size(2)//16)*16,:(self.real_A.size(3)//16)*16] = out
        
        self.pixloss = Variable(self.Tensor([0]))
        self.pixloss = self.backloss_comp(self.fake_B, self.ground_B, self.var_mask)
        self.gradloss = Variable(self.Tensor([0]))
        if self.grad:
            self.gradloss = self.comp_gradloss()
        dif = ((self.fake_B-self.ground_B)/self.ground_B).abs()
        self.relerr = (dif*self.var_mask).sum()/self.var_mask.sum()
        self.netG_A.train()



    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_G(self):
        #print(self.var_mask.data.sum())
        #print(self.real_A.data.shape)
        if self.var_mask.data.sum()>0:
            self.fake_B = self.ground_B.clone()
            self.fake_B.data.zero_()
            self.fake_B[:,:,:,:] = self.netG_A.forward(self.real_A)
            '''
            if self.multiloss:
                self.fake_B_all = self.fake_B
                self.ground_B_fs = self.ground_B
                self.var_mask_fs = self.var_mask

                avgpool = torch.nn.AvgPool2d(2)

                self.fake_B = self.fake_B_all['2']
                self.ground_B = avgpool(self.ground_B_fs)
                self.var_mask = avgpool(self.var_mask_fs)
                self.pixloss_ds2 = self.backloss_comp()

                self.fake_B = self.fake_B_all['1']
                self.ground_B = avgpool(self.ground_B)
                self.var_mask = avgpool(self.var_mask)
                self.pixloss_ds4 = self.backloss_comp()

                self.fake_B = self.fake_B_all['3']
                self.ground_B = self.ground_B_fs
                self.var_mask = self.var_mask_fs
                self.pixloss_fs = self.backloss_comp()

                self.dif = ((self.fake_B-self.ground_B)/self.ground_B).abs()
                self.relerr = (self.dif*self.var_mask).sum()/self.var_mask.sum()

                self.pixloss = self.pixloss_fs + self.multiloss_f*self.pixloss_ds2 + self.multiloss_f*self.pixloss_ds4

                if self.grad:
                    self.gradloss = self.comp_gradloss()
                    self.backloss = self.pixloss + self.gradloss
                else:
                    self.backloss = self.pixloss

                if self.opt.gan:
                    self.gan_loss = self.criterion_D(self.netD_A.forward(torch.cat((self.fake_B,self.ground_B),dim=1)), True)
                    self.backloss = self.backloss + self.opt.gan_lamda_G * self.gan_loss
                self.backloss.backward()
                return
            '''
            # self.loss = ((((self.fake_B-self.ground_B)/self.ground_B)**2)*self.var_mask).sum()/self.var_mask.sum()
            # self.loss = (((self.fake_B-self.ground_B)**2)*self.var_mask).sum()/self.var_mask.sum()
            # self.loss = self.criterion(self.fake_B, self.ground_B)
            
            # self.rmseloss = self.loss**0.5
            

            self.dif = ((self.fake_B-self.ground_B)/self.ground_B).abs()
            self.relerr = (self.dif*self.var_mask).sum()/self.var_mask.sum()
            

            self.pixloss = self.backloss_comp(self.fake_B, self.ground_B, self.var_mask)
            if self.grad:
                self.gradloss = self.comp_gradloss()
                self.backloss = self.pixloss + self.gradloss
            else:
                self.backloss = self.pixloss
            self.backloss.backward()
        else:
            # self.loss = (0*self.real_A).sum()
            print('000000000000000000000000')

    def backward_D(self):
        fake_B = Variable(self.fake_B.data)
        self.D_loss_fake = self.criterion_D(self.netD_A.forward(torch.cat((fake_B,self.ground_B),dim=1)), False)
        self.D_loss_real = self.criterion_D(self.netD_A.forward(torch.cat((self.ground_B,self.ground_B),dim=1)), True)
        self.backloss_D = self.D_loss_fake + self.D_loss_real
        self.backloss_D.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        self.backward_G()
        for optimizer in self.optimizers:
            optimizer.step()

        if self.opt.gan:
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

    def get_current_errors(self):
        if self.multiloss:
            return OrderedDict([('pixloss',self.pixloss_fs.item()),
                ('pixloss_ds2',self.pixloss_ds2.item()),
                ('pixloss_ds4',self.pixloss_ds4.item()),
                ('gradloss',self.gradloss.item()),
                ('relerr',self.relerr.item())])
        return OrderedDict([('pixloss',self.pixloss.item()),('gradloss',self.gradloss.item()),('relerr',self.relerr.item())])
        
    def get_current_visuals(self):
        fake_B = self.fake_B.data.cpu().float().numpy().transpose(3,2,1,0)[:,:,:,0]
        ground_B = self.ground_B.data.cpu().float().numpy().transpose(3,2,1,0)[:,:,:,0]
        mask = self.var_mask.data.cpu().float().numpy().transpose(3,2,1,0)[:,:,:,0]
        image_path = self.image_paths[0]
        start = time.time()
        fake_B = util.inverse_preprocess_tissue_property(fake_B)
        end = time.time()
        util.print_log('Time for denormalizing tissue properties: %.5fs ' % (end-start), self.opt.file_name)
        ground_B = util.inverse_preprocess_tissue_property(ground_B)
        return OrderedDict([('fake_B', fake_B),('ground_B', ground_B), ('mask', mask), ('image_path', image_path)])

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)

    def get_backloss(self):
        if self.criterion == 'L2re':
            self.backloss_comp = lambda fake_B, ground_B, var_mask: ((((fake_B-ground_B)/ground_B)**2)*var_mask).sum()/var_mask.sum()
        elif self.criterion == 'L1re':
            self.backloss_comp = lambda fake_B, ground_B, var_mask: ((((fake_B-ground_B)/ground_B).abs())*var_mask).sum()/var_mask.sum()
        elif self.criterion == 'L1.5re':
            self.backloss_comp = lambda fake_B, ground_B, var_mask: ((dif**1.5)*var_mask).sum()/var_mask.sum()
        elif self.criterion == 'L1ae':
            self.backloss_comp = lambda fake_B, ground_B, var_mask: (((((fake_B-ground_B)).abs())*var_mask).sum()/var_mask.sum())
        elif self.criterion == 'L2ae':
            self.backloss_comp = lambda fake_B, ground_B, var_mask: ((((fake_B-ground_B)**2)*var_mask).sum()/var_mask.sum())
        elif self.criterion == 'L4ae':
            self.backloss_comp = lambda fake_B, ground_B, var_mask: ((((fake_B-ground_B)**4)*var_mask).sum()/var_mask.sum())
        elif self.criterion == 'focal gamma=1':
            self.backloss_comp = lambda fake_B, ground_B, var_mask: (((((fake_B-ground_B)**2/ground_B).abs())*var_mask).sum()/var_mask.sum())
        elif self.criterion == 'anti-fluctuation L1 relative err loss':
            self.backloss_comp = lambda fake_B, ground_B, var_mask: (dif * (dif>0.01).type_as(dif) * var_mask).sum() / ((dif>0.01).type_as(dif) * var_mask).sum()
        elif self.criterion == 'L1ae*gt':
            self.backloss_comp = lambda fake_B, ground_B, var_mask: (((((fake_B-ground_B)*ground_B).abs())*var_mask).sum()/var_mask.sum())
        elif self.criterion == 'L2ae*gt^4':
            self.backloss_comp = lambda fake_B, ground_B, var_mask: ((((((fake_B-ground_B)**2)*(ground_B**4)).abs())*var_mask).sum()/var_mask.sum())
        elif self.criterion == 'L2ae*gt^2':
            self.backloss_comp = lambda fake_B, ground_B, var_mask: ((((((fake_B-ground_B)**2)*(ground_B**2)).abs())*var_mask).sum()/var_mask.sum())
        else:
            raise ValueError('backloss criterion %s not recognized' % self.criterion)

    def get_backloss_new(self, criterion):
        if criterion == 'L2re':
            backloss_comp = lambda fake_B, ground_B, var_mask: ((((fake_B-ground_B)/ground_B)**2)*var_mask).sum()/var_mask.sum()
        elif criterion == 'L1re':
            backloss_comp = lambda fake_B, ground_B, var_mask: ((((fake_B-ground_B)/ground_B).abs())*var_mask).sum()/var_mask.sum()
        elif criterion == 'L1.5re':
            backloss_comp = lambda fake_B, ground_B, var_mask: ((dif**1.5)*var_mask).sum()/var_mask.sum()
        elif criterion == 'L1ae':
            backloss_comp = lambda fake_B, ground_B, var_mask: (((((fake_B-ground_B)).abs())*var_mask).sum()/var_mask.sum())
        elif criterion == 'L2ae':
            backloss_comp = lambda fake_B, ground_B, var_mask: ((((fake_B-ground_B)**2)*var_mask).sum()/var_mask.sum())
        elif criterion == 'L4ae':
            backloss_comp = lambda fake_B, ground_B, var_mask: ((((fake_B-ground_B)**4)*var_mask).sum()/var_mask.sum())
        elif criterion == 'focal gamma=1':
            backloss_comp = lambda fake_B, ground_B, var_mask: (((((fake_B-ground_B)**2/ground_B).abs())*var_mask).sum()/var_mask.sum())
        elif criterion == 'anti-fluctuation L1 relative err loss':
            backloss_comp = lambda fake_B, ground_B, var_mask: (dif * (dif>0.01).type_as(dif) * var_mask).sum() / ((dif>0.01).type_as(dif) * var_mask).sum()
        elif criterion == 'L1ae*gt':
            backloss_comp = lambda fake_B, ground_B, var_mask: (((((fake_B-ground_B)*ground_B).abs())*var_mask).sum()/var_mask.sum())
        elif criterion == 'L2ae*gt^4':
            backloss_comp = lambda fake_B, ground_B, var_mask: ((((((fake_B-ground_B)**2)*(ground_B**4)).abs())*var_mask).sum()/var_mask.sum())
        elif criterion == 'L2ae*gt^2':
            current_index_ibackloss_comp = lambda fake_B, ground_B, var_mask: ((((((fake_B-ground_B)**2)*(ground_B**2)).abs())*var_mask).sum()/var_mask.sum())
        else:
            raise ValueError('backloss criterion %s not recognized' % current_index_icriterion)
        return backloss_comp


    def comp_gradloss(self):
        dif1 = self.fake_B[:,:,:-1,:]-self.fake_B[:,:,1:,:]-(self.ground_B[:,:,:-1,:]-self.ground_B[:,:,1:,:])
        dif2 = self.fake_B[:,:,:,:-1]-self.fake_B[:,:,:,1:]-(self.ground_B[:,:,:,:-1]-self.ground_B[:,:,:,1:])
        difr1 = dif1/(self.ground_B[:,:,:-1,:]+self.ground_B[:,:,1:,:])
        difr2 = dif2/(self.ground_B[:,:,:,:-1]+self.ground_B[:,:,:,1:])
        difr = difr1[:,:,:,:-1].abs() + difr2[:,:,:-1,:].abs()
        mask = self.var_mask[:,:,:-1,:-1] * self.var_mask[:,:,1:,:-1] * self.var_mask[:,:,:-1,1:]
        return (difr*mask).sum()/mask.sum()


