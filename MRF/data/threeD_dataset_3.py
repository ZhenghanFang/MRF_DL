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
        self.n_timepoint = opt.input_nc // opt.multi_slice_n // 2
        self.initialize_base(opt)
        

    def name(self):
        return 'threeD_Dataset_3'
    
    def load_dataset(self, data_path):
        print('load dataset: ', data_path, 'slice:', self.data_args[self.data_index]['slice_i'])
        data = {}
        data['imMRF'] = self.preprocess_imMRF(self.read_imMRF(), flip=self.flipimMRF)
        data['Tmap'] = self.preprocess_Tmap(*self.read_Tmap())
        data['mask'] = self.preprocess_mask(self.read_mask())
        if self.opt.half:
            for k in data:
                data[k] = data[k].astype('float16')
        if self.opt.zerobg:
            data['imMRF'] = data['imMRF'] * data['mask']

        # dataset_path = os.path.splitext(os.path.split(imMRF_path)[-1])[0]
        data['dataset_path'] = self.get_dataset_path(data_path)
        return data
    
    def read_imMRF(self):
        path = self.data_paths[self.data_index]['imMRF']
        slice_i = self.data_args[self.data_index]['slice_i']
        
        # print(type(self.data3D[path]['imMRF'][0:n_timepoint,slice_i:slice_i+self.opt.multi_slice_n]))
        return self.data3D[path]['imMRF'][0:self.n_timepoint,slice_i:slice_i+self.opt.multi_slice_n].copy()
    
    def read_Tmap(self):
        path = self.data_paths[self.data_index]['imMRF']
        slice_i = self.data_args[self.data_index]['slice_i']
        center_slice = (self.opt.multi_slice_n-1) // 2
        return self.data3D[path]['t1'][slice_i + center_slice].copy(), self.data3D[path]['t2'][slice_i + center_slice].copy()
    
    def read_mask(self):
        path = self.data_paths[self.data_index]['imMRF']
        slice_i = self.data_args[self.data_index]['slice_i']
        center_slice = (self.opt.multi_slice_n-1) // 2
        return self.data3D[path]['mask'][slice_i + center_slice].copy()
      
    def preprocess_imMRF(self, imMRF, flip=True):
        # combine slice dimension and time dimension
        imMRF = numpy.reshape(imMRF, (-1, imMRF.shape[2], imMRF.shape[3]), order='F')
        
        if flip:
            # flip to align with ground truth tissue maps
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
    '''
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
    '''
    
    def get_paths(self):
        if self.opt.onMAC: 
            MRF_data_root = '/Users/zhenghanfang/Desktop/standard_MRF/DataNewDictionary'
        else:
            MRF_data_root = '/shenlab/lab_stor/zhenghan/data/MRF/DataNewDictionary'
            
        if self.opt.onMAC:
            d_root = MRF_data_root + '/Data_20190415/3DMRF_prospective/Set1/'
            person_path = ['190324_DLMRF3D_vol1','190324_DLMRF3D_vol2','190328_DLMRF3D_vol3','190330_DLMRF3D_vol4','190330_DLMRF3D_vol5','190407_DLMRF3D_vol6','190407_DLMRF3D_vol7']
            '''
            slice_N_total = [144,176,160,176,176,160,160]
            imMRF_file_name = 'imMRF_GRAPP2_PF_quarterpoints_noSVD.mat'
            Tmap_file_name = 'patternmatching_GRAPPA2_PF_quarterpoints_noSVD.mat'
            mask_file_name = None
            self.flipimMRF = False
            '''
            slice_N_total = [144] * 7
            imMRF_file_name = 'imMRF_GRAPP2_PF_quarterpoints_noSVD.mat'
            Tmap_file_name = 'patternmatching_SVD_PCA.mat'
            mask_file_name = 'immask.mat'
            self.flipimMRF = False
        else:
            # d_root = '/shenlab/lab_stor/zhenghan/data/MRF/3D/'
            # d_root = '/shenlab/lab_stor/zhenghan/3DMRF_noSVD_R3_192pnts/'
            # d_root = '/shenlab/lab_stor/zhenghan/data/MRF/DataNewDictionary/Data_20190307/3DMRF_noSVD_UndersampleOnly_192pnts/'
            # d_root = '/shenlab/lab_stor/zhenghan/data/MRF/DataNewDictionary/Data_20190403/3DMRF_noSVD_GRAPP2_PF_288pnts/'
            # d_root = '/shenlab/lab_stor/zhenghan/data/MRF/DataNewDictionary/Data_20190403/3DMRF_noSVD_GRAPP3_288pnts/'
            # d_root = '/shenlab/lab_stor/zhenghan/data/MRF/DataNewDictionary/Data_20190415/3DMRF_prospective/Set2/'
            d_root = MRF_data_root + '/Data_3DMRF/3DMRF_noSVD_40slices/'
            person_path = ['190330_DLMRF3D_vol4','190330_DLMRF3D_vol5','190407_DLMRF3D_vol6']
            slice_N_total = [40,40,40]
            imMRF_file_name = 'imMRF_AF2_PF_allpoints_noSVD.mat'
            Tmap_file_name = 'patternmatching_noSVD.mat'
            mask_file_name = 'mask.mat'
            self.flipimMRF = True
            
        
        # person_path = ['1_180410','2_180603','3_180722','4_180812_1','5_180812_2']
        # person_path = ['180408','180603','180722','180812_1','180812_2']
        # slice_N = [94,94,94,94,94]
        # slice_N = [1,1,1,1,1]
        
        slice_N = [x - (self.opt.multi_slice_n-1) for x in slice_N_total]
        test_i = self.opt.test_i
        if self.opt.set_type == 'train':
            if test_i == -1:
                person = list(range(0,len(slice_N)))
            else:
                person = list(range(0,test_i))+list(range(test_i+1,len(slice_N)))
        else:
            if test_i == -1:
                person = []
            else:
                person = list(range(test_i,test_i+1))
        
        # For fast debugging:
        # person = person[0:1]

        self.data_paths = []
        self.data_args = []
        self.data3D = {}
        for i in range(len(person)):
            imMRF_path = d_root+person_path[person[i]]+'/'+imMRF_file_name
            Tmap_path  = d_root+person_path[person[i]]+'/'+Tmap_file_name
            mask_path  = d_root+person_path[person[i]]+'/'+mask_file_name if mask_file_name else None
            for j in range(slice_N[person[i]]):
                self.data_paths.append({
                    'imMRF': imMRF_path,
                    'Tmap':  Tmap_path,
                    'mask':  mask_path
                    })
                self.data_args.append({'slice_i': j})
            print('loading data:', imMRF_path)
            self.data3D[imMRF_path] = {}
            self.data3D[imMRF_path]['imMRF'] = h5py.File(imMRF_path, 'r')['imMRF_all'][0:self.n_timepoint]
            self.data3D[imMRF_path]['t1'] = h5py.File(Tmap_path, 'r')['t1big_all'][:]
            self.data3D[imMRF_path]['t2'] = h5py.File(Tmap_path, 'r')['t2big_all'][:]
            if not mask_file_name:
                self.data3D[imMRF_path]['mask'] = self.data3D[imMRF_path]['t1'] * 0.0 + 1.0
            else:
                self.data3D[imMRF_path]['mask'] = h5py.File(mask_path, 'r')['immask'][:]
