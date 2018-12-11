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
            self.imMRF_paths = [
            '/shenlab/lab_stor/zhenghan/data/MRF/3DMRF_2/1_180410/',
            '/shenlab/lab_stor/zhenghan/data/MRF/3DMRF_2/2_180603/',
            # '/shenlab/lab_stor/zhenghan/data/MRF/3DMRF_2/3_180722/',
            '/shenlab/lab_stor/zhenghan/data/MRF/3DMRF_2/4_180812_1/',
            '/shenlab/lab_stor/zhenghan/data/MRF/3DMRF_2/5_180812_2/'
            ]
        elif self.set_type == 'val':
            # self.imMRF_paths = ['/shenlab/lab_stor/zhenghan/data/MRF/3DMRF_2/2_180603/']
            self.imMRF_paths = ['/shenlab/lab_stor/zhenghan/data/MRF/3DMRF_2/3_180722/']

        if self.set_type == 'val':
            self.load_data()

        if self.set_type == 'train':
            self.patchSize = opt.patchSize
            self.current_index_i = 0



    def __getitem__(self, index):
        if self.set_type == 'val':
            subject_i = index // (96 - self.opt.multi_slice_n + 1)
            slice_i = index % (96 - self.opt.multi_slice_n + 1) + int((self.opt.multi_slice_n-1)/2)
            patchSize = 256
            patch_i_1, patch_i_2 = 0, 0

        elif self.set_type == 'train':
            patchSize = self.patchSize
            subject_i = random.randint(0, len(self.data) - 1)
            time_start = time.time()
            while True:
                slice_i = random.randint((self.opt.multi_slice_n-1)/2, 96 - (self.opt.multi_slice_n-1)/2 - 1)
                patch_i_1, patch_i_2 = ( random.randint(0, 256 - patchSize), 
                    random.randint(0, 256 - patchSize) )

                mask = ( self.data[subject_i]['mask']
                    [slice_i:slice_i+1, patch_i_1:patch_i_1+patchSize, patch_i_2:patch_i_2+patchSize] )
                if mask.sum() > 0.5 * mask.size:
                    break

        mask = ( self.data[subject_i]['mask']
            [slice_i:slice_i+1, patch_i_1:patch_i_1+patchSize, patch_i_2:patch_i_2+patchSize] )

        input_G = self.data[subject_i]['imMRF'][:,slice_i - int((self.opt.multi_slice_n-1)/2):slice_i + int((self.opt.multi_slice_n-1)/2) + 1,patch_i_1:patch_i_1+patchSize, patch_i_2:patch_i_2+patchSize]
        input_G = numpy.concatenate((input_G['real'],input_G['imag']), axis=0)
        input_G = numpy.concatenate(numpy.split(input_G, input_G.shape[1], axis=1), axis=0)
        input_G = input_G.squeeze()
        t = numpy.mean(input_G ** 2, axis=0) * 2
        t = t[numpy.newaxis,:,:]
        input_G = input_G / (t**0.5)
        input_G = input_G / 36

        label_G = numpy.concatenate((
            self.data[subject_i]['T1map']
            [slice_i:slice_i+1, patch_i_1:patch_i_1+patchSize, patch_i_2:patch_i_2+patchSize] / 5000,
            self.data[subject_i]['T2map']
            [slice_i:slice_i+1, patch_i_1:patch_i_1+patchSize, patch_i_2:patch_i_2+patchSize] / 500
            ), axis=0)


        input_G = torch.from_numpy(input_G.copy()).float()
        label_G = torch.from_numpy(label_G.copy()).float()
        mask = torch.from_numpy(mask.copy()).float()

        

        return {'A': input_G, 'B': label_G, 'mask': mask, 'A_paths': self.data[subject_i]['subject_path'] + '_s' + str(slice_i)}


    def __len__(self):
        if self.set_type == 'train':
            return len(self.data) * int(256*256/16/16*96)
        elif self.set_type == 'val':
            return len(self.data) * (96 - self.opt.multi_slice_n + 1)




    def name(self):
        return 'MRFDataset'

    def load_data(self):
        if self.set_type == 'val':
            imMRF_paths = self.imMRF_paths
        elif self.set_type == 'train':
            if self.current_index_i == 0:
                self.index = list(range(len(self.imMRF_paths)))
                random.shuffle(self.index)
            imMRF_paths = []
            for i in range(2):
                imMRF_paths.append(self.imMRF_paths[self.index[self.current_index_i]])
                self.current_index_i = self.current_index_i + 1
                if self.current_index_i == len(self.index):
                    self.current_index_i = 0
                    break

        self.data = []

        for imMRF_path in imMRF_paths:
            dir_path = imMRF_path
            imMRF_path = dir_path + 'imMRF_GRAPPA_Rz2_PF_256point.mat'
            print('load data: ', imMRF_path)

            file = h5py.File(imMRF_path, 'r')
            
            imMRF = file['imMRFc'][:]
            imMRF = numpy.flip(numpy.flip(imMRF,2),3)
            print("load imMRF")
            gt_path = dir_path + 'patternmatching_SVD_PCA.mat'
            file = h5py.File(gt_path, 'r')
            T1map = file['t1bigc_all'][:]
            T2map = file['t2bigc_all'][:]
            mask_path = dir_path + 'immask_h5.mat'
            file = h5py.File(mask_path, 'r')
            mask = file['immask'][:]
            subject_path = os.path.split(imMRF_path)[0].split(os.sep)[-1]

            # imMRF = file['imMRF'][:]
            # print("load imMRF")
            # T1map = file['t1big'][:]
            # T2map = file['t2big'][:]
            # mask = file['mask'][:]
            # subject_path = os.path.splitext(os.path.split(imMRF_path)[-1])[0]


            self.data.append({'imMRF':imMRF, 'T1map': T1map, 'T2map': T2map, 'mask': mask, 'subject_path': subject_path})
        
