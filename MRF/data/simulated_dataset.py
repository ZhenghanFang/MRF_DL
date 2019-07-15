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
        self.initialize_base(opt)

    def name(self):
        return 'simulated_Dataset'

    def read_mask(self, file):
        return file['immask'][:]
    
    
    def preprocess_imMRF(self, imMRF, flip=True):
        if flip:
            # preprocess with flipping to align with ground truth tissue maps
            imMRF = imMRF[:, ::-1, ::-1]
        # imMRF = numpy.flip(numpy.flip(imMRF,1),2)
        A_img = imMRF
        A_img = numpy.concatenate((A_img['real'], A_img['imag']), axis=0).astype('float32')
        
        # flip the real part of half of slices
        print(A_img)
        if random.random() > 0.5:
            A_img[0] = - A_img[0]
        print(A_img)
        
        # normalization
        if self.opt.data_norm == 'non':
            print("no normalization")
        else:
            A_img = A_img * 1e-5
            t = numpy.mean(A_img ** 2, axis=0) * 2
            A_img = A_img / (t[numpy.newaxis,:,:] ** 0.5) / 36
        return A_img
    
    
    def get_paths(self):
        if self.opt.onMAC:
            d_root = '/Users/zhenghanfang/Desktop/standard_MRF/DataNewDictionary/Data_20181017/SimulatedTrainingData/'
        else:
            d_root = '/shenlab/lab_stor/zhenghan/data/MRF/simulate/SimulatedTrainingData/'
        person_path = ['181007', '181012-1', '181012-2', '181014-1', '181014-2']
        slice_path = [
            [str(i) for i in range(110,129,2)],
            [str(i) for i in range(106,125,2)],
            [str(i) for i in range(136,155,2)],
            [str(i) for i in range(123,142,2)],
            [str(i) for i in range(155,174,2)]
        ]
        slice_N = [10, 10, 10, 10, 10]
        # slice_N = [1,1,1,1,1,1]
        test_i = 4
        if self.set_type == 'train':
            person = list(range(0,test_i))+list(range(test_i+1,5))
        elif self.set_type == 'val':
            person = list(range(test_i,test_i+1))
            
        self.data_paths = []
        for i in range(len(person)):
            for j in range(slice_N[person[i]]):
                d_path = d_root + person_path[person[i]] + '/' + slice_path[person[i]][j] + '/'
                self.data_paths.append({
                    'imMRF': d_path + 'simu_imMRF.mat',
                    'Tmap': d_path + 'simu_patternmatching.mat',
                    'mask': d_path + 'simu_immask_resize.mat'
                    })

        # if self.set_type == 'val':
        #     if self.opt.onMAC:
        #         d_root = '/Users/zhenghanfang/Desktop/standard_MRF/DataNewDictionary/'
        #     else:
        #         d_root = '/shenlab/lab_stor/zhenghan/data/MRF/DataNewDictionary/'
        #     person_path = ['20180206data/180114','20180206data/180124','20180206data/180131_1','20180206data/180131_2','20180206data/180202','newMRFData_041218/180408_1','newMRFData_041218/180408_2']
        #     slice_N = [12,12,12,12,12,10,12]
        #     # slice_N = [1,1,1,1,1,1,1]
        #     test_i = 5
        #     person = list(range(test_i,test_i+1))
                
        #     self.data_paths = []
        #     for i in range(len(person)):
        #         a = os.listdir(d_root+person_path[person[i]-1])
        #         for p in a:
        #             if p[0] == '.':
        #                 a.remove(p)
        #         for j in range(slice_N[person[i]-1]):
        #             self.data_paths.append({
        #             'imMRF': d_root+person_path[person[i]-1]+'/'+a[j]+'/imMRF.mat',
        #             'Tmap': d_root+'PatternMatching_2304/sub_'+str(person[i])+'/'+str(j+1)+'/patternmatching.mat',
        #             'mask': '/Users/zhenghanfang/Desktop/standard_MRF/'+'Data_Qian_skull_h5/'+str(person[i])+'/'+str(j+1)+'-skull.mat'
        #             })
