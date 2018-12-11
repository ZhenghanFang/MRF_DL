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

class MRFDataset(BaseDataset):
    def initialize(self, opt):
        self.flipimMRF = False
        self.initialize_base(opt)
        
    def name(self):
        return 'dict_Dataset'

    def transform_train(self, sample):
        sample['input_G'] = sample['input_G'] * (1 + 0.01 * numpy.random.standard_normal(sample['input_G'].shape))
        return sample

    def get_paths(self):
        if self.opt.onMAC:
            d_root = '/Users/zhenghanfang/Desktop/standard_MRF/DataNewDictionary/'
        else:
            d_root = '/shenlab/lab_stor/zhenghan/data/MRF/DataNewDictionary/'
        person_path = ['20180206data/180114','20180206data/180124','20180206data/180131_1','20180206data/180131_2','20180206data/180202','newMRFData_041218/180408_1','newMRFData_041218/180408_2']
        slice_path = [
            [str(i) for i in range(181,206,2)],
            [str(i) for i in range(44,67,2)],
            [str(i) for i in range(43,66,2)],
            [str(i) for i in range(87,110,2)],
            [str(i) for i in range(154,177,2)],
            [str(i) for i in range(90,109,2)],
            [str(i) for i in range(120,143,2)]
        ]
        slice_N = [12,12,12,12,12,10,12]
        # slice_N = [1,1,1,1,1,1,1]
        test_i = self.opt.test_i
        if self.set_type == 'train':
            person = list(range(1,test_i))+list(range(test_i+1,7))
        else:
            person = list(range(test_i,test_i+1))
        # person = list(range(1,7))

        self.data_paths = []
        for i in range(len(person)):
            for j in range(slice_N[person[i]-1]):
                d_path = d_root + person_path[person[i]-1] + '/' + slice_path[person[i]-1][j] + '/'
                mask_path = d_root+'mask_large/sub_'+str(person[i])+'/'+str(j+1)+'.mat' # self.mask_type == 'large'
                # mask_path = d_root+'Data_Qian_skull_h5/'+str(person[i])+'/'+str(j+1)+'-skull.mat' # small mask
                Tmap_path = d_root+'PatternMatching_2304/sub_'+str(person[i])+'/'+str(j+1)+'/patternmatching.mat'
                self.data_paths.append({
                    'imMRF': d_path+'imMRF_dict.mat',
                    'Tmap': Tmap_path,
                    'mask': mask_path,
                    })
