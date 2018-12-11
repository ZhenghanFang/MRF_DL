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
        self.set_type = opt.set_type

        if self.set_type == 'train':
            self.imMRF_paths = util.scan_files('/shenlab/lab_stor/zhenghan/data/MRF/residue/h5', ['180114','180124','180131_1','180131_2','180408_1'])
        elif self.set_type == 'val':
            self.imMRF_paths = util.scan_files('/shenlab/lab_stor/zhenghan/data/MRF/residue/h5', ['180202'])
            
        if self.set_type == 'val':
            self.load_data()

        if self.set_type == 'train':
            self.patchSize = opt.patchSize
            self.current_index_i = 0
        

    def __getitem__(self, index):
        dataset_i = index % len(self.data)
        

        if self.set_type == 'val':
            patchSize = 256
            patch_i_1, patch_i_2 = 0, 0

        elif self.set_type == 'train':
            patchSize = self.patchSize
            time_start = time.time()
            while True:
                patch_i_1, patch_i_2 = ( random.randint(0, 256 - patchSize), 
                    random.randint(0, 256 - patchSize) )
                mask = ( self.data[dataset_i]['mask']
                    [patch_i_1:patch_i_1+patchSize, patch_i_2:patch_i_2+patchSize] )
                if mask.sum() > 0.5 * mask.size:
                    break

        mask = ( self.data[dataset_i]['mask']
            [numpy.newaxis, patch_i_1:patch_i_1+patchSize, patch_i_2:patch_i_2+patchSize] )

        input_G = self.data[dataset_i]['imMRF'][:, patch_i_1:patch_i_1+patchSize, patch_i_2:patch_i_2+patchSize]
        # t = numpy.mean(input_G ** 2, axis=0) * 2
        # t = t[numpy.newaxis,:,:]
        # input_G = input_G / (t**0.5)
        # input_G = input_G / 36

        label_G = numpy.concatenate((
            self.data[dataset_i]['T1map']
            [numpy.newaxis, patch_i_1:patch_i_1+patchSize, patch_i_2:patch_i_2+patchSize] / 200,
            self.data[dataset_i]['T2map']
            [numpy.newaxis, patch_i_1:patch_i_1+patchSize, patch_i_2:patch_i_2+patchSize] / 10
            ), axis=0)


        input_G = torch.from_numpy(input_G)
        label_G = torch.from_numpy(label_G)
        mask = torch.from_numpy(mask)

        

        return {'A': input_G, 'B': label_G, 'mask': mask, 'A_paths': self.data[dataset_i]['dataset_path']}


    def __len__(self):
        if self.set_type == 'train':
            return len(self.data) * int(256*256/32/32) * 2
        elif self.set_type == 'val':
            return len(self.data)



    def name(self):
        return 'Residue_Dataset'


    def load_data(self):
        if self.set_type == 'val':
            imMRF_paths = self.imMRF_paths
        elif self.set_type == 'train':
            if self.current_index_i == 0:
                self.index = list(range(len(self.imMRF_paths)))
                random.shuffle(self.index)
            imMRF_paths = []
            for i in range(29):
                imMRF_paths.append(self.imMRF_paths[self.index[self.current_index_i]])
                self.current_index_i = self.current_index_i + 1
                if self.current_index_i == len(self.index):
                    self.current_index_i = 0
                    break

        self.data = []

        for imMRF_path in imMRF_paths:
            print('load data: ', imMRF_path)

            file = h5py.File(imMRF_path, 'r')
            
            imMRF = file['imMRF_res'][:]
            print("load imMRF")
            T1map = file['t1big_res'][:]
            T2map = file['t2big_res'][:]
            mask = file['mask'][:]
            dataset_path = os.path.splitext(os.path.split(imMRF_path)[-1])[0]


            self.data.append({'imMRF':imMRF, 'T1map': T1map, 'T2map': T2map, 'mask': mask, 'dataset_path': dataset_path})
        
