# import os.path
# import torchvision.transforms as transforms
# from data.base_dataset import BaseDataset, get_transform
from data.base_dataset import BaseDataset
# from data.image_folder import make_dataset
# from PIL import Image
# import PIL
import h5py
import random
import torch
import numpy
import math
import skimage.transform
import time
import scipy.io as sio
import os
import util.util as util
import time

class MRFDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt

        if opt.set_type == 'val':
            imMRF_paths = util.scan_files('/shenlab/lab_stor/zhenghan/data/MRF/motion/val_fix_h5')
            self.patchSize = 0
        elif opt.set_type == 'train':
            imMRF_paths = util.scan_files('/shenlab/lab_stor/zhenghan/data/MRF/motion/train_fix_h5_2')
            self.patchSize = opt.patchSize
        print(imMRF_paths)
        data = []
        for imMRF_path in imMRF_paths:
            file = h5py.File(imMRF_path, 'r')
            imMRF = file['imMRF'][:]
            print("load imMRF")
            
            T1map = file['t1big'][:]
            T2map = file['t2big'][:]
            mask = file['mask'][:]

            data.append({'imMRF':imMRF, 'T1map': T1map, 'T2map': T2map, 'mask': mask})
        
        self.data = data
        self.train_path = '/shenlab/lab_stor/zhenghan/data/MRF/motion/train_fix_h5_2'
        # print(self.data)
        



    def __getitem__(self, index):
        patchSize = self.patchSize
        subject_i = index % len(self.data)
        # print("subject_i = ", subject_i)
        if patchSize == 0:
            patchSize = 256

        time_start = time.time()

        while True:
            patch_i_1, patch_i_2 = (random.randint(0, 256 - patchSize), 
                random.randint(0, 256 - patchSize))
            mask = (self.data[subject_i]['mask']
                [patch_i_1:patch_i_1+patchSize, patch_i_2:patch_i_2+patchSize])
            if mask.sum() > 0.5 * mask.size or self.opt.set_type == 'val':
                break

        


        
        input_G = self.data[subject_i]['imMRF'][:,patch_i_1:patch_i_1+patchSize, patch_i_2:patch_i_2+patchSize]
        # print("load imMRF, %.2f" % (time.time()-time_start))
        
        label_G = numpy.stack(
            (self.data[subject_i]['T1map']
            [patch_i_1:patch_i_1+patchSize, patch_i_2:patch_i_2+patchSize] / 5000, 
            self.data[subject_i]['T2map']
            [patch_i_1:patch_i_1+patchSize, patch_i_2:patch_i_2+patchSize] / 500)
            , axis=0)
        

        mask = numpy.expand_dims(mask, axis=0)

        input_G = torch.from_numpy(input_G)
        label_G = torch.from_numpy(label_G)
        mask = torch.from_numpy(mask)

        return {'A': input_G, 'B': label_G, 'mask': mask, 'A_paths': ''}

    def __len__(self):
        if self.patchSize == 0:
            return len(self.data)
        else:
            return len(self.data) * 100


    def name(self):
        return 'MRFDataset'

    def get_patch_pos(self, opt):
        self.change_dataset(opt)

    def change_dataset(self, opt):

        if self.train_path == '/shenlab/lab_stor/zhenghan/data/MRF/motion/train_fix_h5_2':
            self.train_path = '/shenlab/lab_stor/zhenghan/data/MRF/motion/train_fix_h5'
        else:
            self.train_path = '/shenlab/lab_stor/zhenghan/data/MRF/motion/train_fix_h5_2'
        
        imMRF_paths = util.scan_files(self.train_path, '')
            
        # print(imMRF_paths)
        data = []
        for imMRF_path in imMRF_paths:
            file = h5py.File(imMRF_path, 'r')
            imMRF = file['imMRF'][:]
            print("load imMRF")
            
            T1map = file['t1big'][:]
            T2map = file['t2big'][:]
            mask = file['mask'][:]

            data.append({'imMRF':imMRF, 'T1map': T1map, 'T2map': T2map, 'mask': mask})
        
        self.data = data


