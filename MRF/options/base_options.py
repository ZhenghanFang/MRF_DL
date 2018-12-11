import argparse
import os
from util import util
import torch
import subprocess

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
    #    self.parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--batchSize', type=int, default=2, help='input batch size')
    #    self.parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
    #    self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--input_nc', type=int, default=192*2*3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    #    self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
    #    self.parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
        self.parser.add_argument('--which_model_netG', type=str, default='UniNet_init', help='selects model to use for netG')
    #    self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
    #    self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='3D', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--dataset', type=str, default='single_dataset', help='chooses how datasets are loaded. [mrf_dataset | single_dataset]')
    #    self.parser.add_argument('--model', type=str, default='cycle_gan',
    #                             help='chooses which model to use. cycle_gan, pix2pix, test')
    #    self.parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='automatic', help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='batch', help='instance normalization or batch normalization')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
    #    self.parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
    #    self.parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')
    #    self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
    #    self.parser.add_argument('--display_single_pane_ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.parser.add_argument('--no_dropout', default=True, action='store_true', help='no dropout for the generator')
    #    self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
    #    self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
    #    self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
    #    self.parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
        
        self.parser.add_argument('--data_GPU', action='store_true', help='save data on GPU')
        self.parser.add_argument('--goal_type', type=str, default='T1T2', help='Goal type (T1 or T2)')
        self.parser.add_argument('--patchSize', type=int, default=32, help='patch length & width')
        self.parser.add_argument('--PCA_n', type=int, default=17, help='number of eigen vectors in PCA')
        self.parser.add_argument('--PCA', action='store_true', help='use PCA')
        self.parser.add_argument('--at', action='store_true', help='Auto-context model')
        self.parser.add_argument('--data_norm', type=str, default='energy', help='data normalization method')
        self.parser.add_argument('--mask', type=str, default='tight', help='mask type')
        self.parser.add_argument('--patchStride', type=int, default=32, help='patch stride')
        self.parser.add_argument('--new_data_format', action='store_true', help='use new data format')
        self.parser.add_argument('--PreNetwork_path', type=str, default='tight', help='mask type')
        self.parser.add_argument('--temp', type=int, default=1, help='temporary option')
        self.parser.add_argument('--FNN_depth', type=int, default=4, help='depth of FNN')
        self.parser.add_argument('--FNN_decrease', type=int, default=0, help='FNN features decrease by layer')
        self.parser.add_argument('--Unet_struc', type=str, default='3ds', help='Unet structure')
        self.parser.add_argument('--test_i', type=int, default=5, help='1~6, index of test subject')
        self.parser.add_argument('--onMAC', action='store_true', help='Run on iMAC')
        self.parser.add_argument('--multiloss', action='store_true', help='multi-scale loss')
        self.parser.add_argument('--multiloss_f', type=float, default=1.0, help='factor of multiloss')
        self.parser.add_argument('--magnitude', action='store_true', help='only input magnitude')
        self.parser.add_argument('--multi_slice_n', type=int, default=3, help='number of slices as input (for 3D data)')



        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        '''
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])
        '''
        if not self.opt.onMAC:
            host = subprocess.check_output('hostname').decode('utf-8')[:-1]
        
            if host == 'stilson' or host == 'andrew' or host == 'wiggin':
                self.opt.checkpoints_dir = '/raid/zhenghan/checkpoints'
            elif host == 'badin' or host == 'bogue' or host == 'burgaw':
                self.opt.checkpoints_dir = '/shenlab/local/zhenghan/checkpoints'
            else:
                raise ValueError("cannot decide checkpoints_dir, server '%s' not recognized." % host)
        
        # if data is loaded on GPU, nThreads should be 0 and gpu_ids must not be []
        if self.opt.data_GPU:
            self.opt.nThreads = 0
            # assert len(self.opt.gpu_ids) > 0, "set data_GPU as true but gpu_ids is []"
        
        if self.opt.goal_type=='T1&T2':
            assert self.opt.output_nc == 2
        
        if self.opt.which_model_netG == 'simple_conv_small_PCA':
            self.opt.PCA = True

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        if self.isTrain:
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        else:
            expr_dir = os.path.join(self.opt.results_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt_'+self.opt.phase+'.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
