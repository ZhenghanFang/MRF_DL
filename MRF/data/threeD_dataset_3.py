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
# import skimage.transform
import time
import scipy.io as sio
import os
import util.util as util
import time

class MRFDataset(BaseDataset):
    def initialize(self, opt):
        self.flipimMRF = False
        self.opt = opt
        self.set_type = opt.set_type
        self.device = torch.device('cuda' if self.opt.gpu_ids else 'cpu')

        if opt.isTrain:
            self.augmentation = opt.augmentation
        else:
            self.augmentation = False

        self.get_paths()

    def name(self):
        return 'threeD_Dataset_3'
    
    def read_imMRF(self, file):
        slice_i = self.data_args[self.data_index]['slice_i']
        n_timepoint = self.opt.input_nc // self.opt.multi_slice_n // 2
        return file['imMRF_all'][0:n_timepoint,slice_i:slice_i+self.opt.multi_slice_n]
      
    def read_Tmap(self, file):
        slice_i = self.data_args[self.data_index]['slice_i']
        center_slice = (self.opt.multi_slice_n-1) // 2
        return file['t1big_all'][slice_i + center_slice], file['t2big_all'][slice_i + center_slice]
    
    def read_mask(self, file):
        slice_i = self.data_args[self.data_index]['slice_i']
        center_slice = (self.opt.multi_slice_n-1) // 2
        return file['m0big_all'][slice_i + center_slice]
      
    def preprocess_imMRF(self, imMRF, flip=True):
        # combine slice dimension and time dimension
        imMRF = numpy.reshape(imMRF, (-1, imMRF.shape[2], imMRF.shape[3]), order='F')
        
        if flip:
            # preprocess with flipping to align with ground truth tissue maps
            imMRF = imMRF[:, ::-1, ::-1]
        # imMRF = numpy.flip(numpy.flip(imMRF,1),2)
        A_img = imMRF
        A_img = numpy.concatenate((A_img['real'], A_img['imag']), axis=0).astype('float32')
        
        # normalization
        if self.opt.data_norm == 'non':
            print("no normalization")
        else:
            t = numpy.mean(A_img ** 2, axis=0) * 2
            A_img = A_img / (t[numpy.newaxis,:,:] ** 0.5) / 36
        return A_img
    
    def __getitem__(self, index):
        self.data_index = index % len(self.data_paths)
        data = self.load_dataset(self.data_paths[self.data_index])

        if self.set_type == 'val':
            sample = {}
            sample['input_G'], sample['label_G'], sample['mask'] = (
                data['imMRF'],
                data['Tmap'],
                data['mask']
                )
            sample = self.np2Tensor(sample)
        return {'A': sample['input_G'], 'B': sample['label_G'], 'mask': sample['mask'], 'A_paths': self.data_paths[self.data_index]['imMRF']}
    
    def __len__(self):
        return len(self.data_paths)
    
    def get_paths(self):
        if self.opt.onMAC:
            # d_root = '/Users/zhenghanfang/Desktop/standard_MRF/DataNewDictionary/Data_20180822/3DMRF/'
            d_root = '/Users/zhenghanfang/Desktop/standard_MRF/DataNewDictionary/Data_20190415/3DMRF_prospective/Set2/'
        else:
            # d_root = '/shenlab/lab_stor/zhenghan/data/MRF/3D/'
            # d_root = '/shenlab/lab_stor/zhenghan/3DMRF_noSVD_R3_192pnts/'
            # d_root = '/shenlab/lab_stor/zhenghan/data/MRF/DataNewDictionary/Data_20190307/3DMRF_noSVD_UndersampleOnly_192pnts/'
            # d_root = '/shenlab/lab_stor/zhenghan/data/MRF/DataNewDictionary/Data_20190403/3DMRF_noSVD_GRAPP2_PF_288pnts/'
            # d_root = '/shenlab/lab_stor/zhenghan/data/MRF/DataNewDictionary/Data_20190403/3DMRF_noSVD_GRAPP3_288pnts/'
            d_root = '/shenlab/lab_stor/zhenghan/data/MRF/DataNewDictionary/Data_20190415/3DMRF_prospective/Set2/'
        # person_path = ['1_180410','2_180603','3_180722','4_180812_1','5_180812_2']
        # person_path = ['180408','180603','180722','180812_1','180812_2']
        person_path = ['190324_DLMRF3D_vol1','190324_DLMRF3D_vol2','190328_DLMRF3D_vol3','190330_DLMRF3D_vol4','190330_DLMRF3D_vol5','190407_DLMRF3D_vol6','190407_DLMRF3D_vol7']
        # slice_N = [94,94,94,94,94]
        # slice_N = [1,1,1,1,1]
        slice_N = [144,176,160,176,176,160,160]
        slice_N[:]=[x - (self.opt.multi_slice_n-1) for x in slice_N]
        test_i = self.opt.test_i
        if self.opt.set_type == 'train':
            person = list(range(0,test_i))+list(range(test_i+1,5))
        else:
            person = list(range(test_i,test_i+1))

        self.data_paths = []
        self.data_args = []
        for i in range(len(person)):
            for j in range(slice_N[person[i]]):
                self.data_paths.append({
                    'imMRF': d_root+person_path[person[i]]+'/imMRF_GRAPP2_PF_quarterpoints_noSVD.mat',
                    'Tmap':  d_root+person_path[person[i]]+'/patternmatching_GRAPPA2_PF_quarterpoints_noSVD.mat',
                    'mask':  d_root+person_path[person[i]]+'/patternmatching_GRAPPA2_PF_quarterpoints_noSVD.mat'
                    })
                self.data_args.append({'slice_i': j})
