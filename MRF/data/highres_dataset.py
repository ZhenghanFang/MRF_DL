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
        self.flipimMRF = True
        self.initialize_base(opt)
        
    def name(self):
        return 'highres_Dataset'

    def get_dataset_path(self, data_path):
        return os.path.split(data_path['imMRF'])[-2:]

    def get_paths(self):
        if self.opt.onMAC:
            d_root = '/Users/zhenghanfang/Desktop/standard_MRF/DataNewDictionary/Data_20181017/Highres/'
        else:
            d_root = '/shenlab/lab_stor/zhenghan/data/MRF/highres/Highres/'
        person_path = ['180923_2', '181007', '181012-1', '181012-2', '181014-1', '181014-2']
        slice_path = [
            ['192', '199', '205', '211'],
            [str(i) for i in range(110,129,2)],
            [str(i) for i in range(106,125,2)],
            [str(i) for i in range(136,155,2)],
            [str(i) for i in range(123,142,2)],
            [str(i) for i in range(155,174,2)]
        ]
        slice_N = [4, 10, 10, 10, 10, 10]
        # slice_N = [1,1,1,1,1,1]
        test_i = 5
        if self.set_type == 'train':
            person = list(range(0,test_i))+list(range(test_i+1,6))
        elif self.set_type == 'val':
            person = list(range(test_i,test_i+1))
            
        self.data_paths = []
        for i in range(len(person)):
            for j in range(slice_N[person[i]]):
                d_path = d_root + person_path[person[i]] + '/' + slice_path[person[i]][j] + '/'
                self.data_paths.append({
                    'imMRF': d_path + 'imMRF_use1st.mat',
                    'Tmap': d_path + 'patternmatching_multishot.mat',
                    'mask': d_path + 'mask.mat'
                    })
