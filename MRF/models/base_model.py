import os
import torch


class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        if self.isTrain:
            self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.device = torch.device('cuda' if self.gpu_ids else 'cpu')

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(device=gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self, network, save_path):
        # save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        # save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))
    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

'''
    def get_backloss_old(self):
        if self.criterion == 'L2re':
            self.backloss_comp = lambda : ((((self.fake_B-self.ground_B)/self.ground_B)**2)*self.var_mask).sum()/self.var_mask.sum()
        elif self.criterion == 'L1re':
            self.backloss_comp = lambda : ((((self.fake_B-self.ground_B)/self.ground_B).abs())*self.var_mask).sum()/self.var_mask.sum()
        elif self.criterion == 'L1.5re':
            self.backloss_comp = lambda : ((self.dif**1.5)*self.var_mask).sum()/self.var_mask.sum()
        elif self.criterion == 'L1ae':
            self.backloss_comp = lambda : (((((self.fake_B-self.ground_B)).abs())*self.var_mask).sum()/self.var_mask.sum())
        elif self.criterion == 'L2ae':
            self.backloss_comp = lambda : ((((self.fake_B-self.ground_B)**2)*self.var_mask).sum()/self.var_mask.sum())
        elif self.criterion == 'L4ae':
            self.backloss_comp = lambda : ((((self.fake_B-self.ground_B)**4)*self.var_mask).sum()/self.var_mask.sum())
        elif self.criterion == 'focal gamma=1':
            self.backloss_comp = lambda : (((((self.fake_B-self.ground_B)**2/self.ground_B).abs())*self.var_mask).sum()/self.var_mask.sum())
        elif self.criterion == 'anti-fluctuation L1 relative err loss':
            self.backloss_comp = lambda : (self.dif * (self.dif>0.01).type_as(self.dif) * self.var_mask).sum() / ((self.dif>0.01).type_as(self.dif) * self.var_mask).sum()
        elif self.criterion == 'L1ae*gt':
            self.backloss_comp = lambda : (((((self.fake_B-self.ground_B)*self.ground_B).abs())*self.var_mask).sum()/self.var_mask.sum())
        elif self.criterion == 'L2ae*gt^4':
            self.backloss_comp = lambda : ((((((self.fake_B-self.ground_B)**2)*(self.ground_B**4)).abs())*self.var_mask).sum()/self.var_mask.sum())
        elif self.criterion == 'L2ae*gt^2':
            self.backloss_comp = lambda : ((((((self.fake_B-self.ground_B)**2)*(self.ground_B**2)).abs())*self.var_mask).sum()/self.var_mask.sum())
        else:
            raise ValueError('backloss criterion %s not recognized' % self.criterion)
'''
