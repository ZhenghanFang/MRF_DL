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
        return 'simulated_SVD_Dataset'
    
    
    def preprocess_imMRF(self, imMRF, flip=True):
        if flip:
            # preprocess with flipping to align with ground truth tissue maps
            imMRF = imMRF[:, ::-1, ::-1]
        # imMRF = numpy.flip(numpy.flip(imMRF,1),2)
        A_img = imMRF
        
        # normalization
        if self.opt.data_norm == 'non':
            print("no normalization")
        else:
            t = numpy.mean(A_img ** 2, axis=0) * 2
            A_img = A_img / (t[numpy.newaxis,:,:] ** 0.5) / 36
        return A_img
    
    
    def get_paths(self):
        if self.opt.onMAC:
            MRF_data_root = '/Users/zhenghanfang/Desktop/standard_MRF/DataNewDictionary'
        else:
            MRF_data_root = '/shenlab/lab_stor/zhenghan/data/MRF/DataNewDictionary'
        data_root = MRF_data_root + '/Data_simu/SVD_flip'
        
        test_i = [10] + list(range(26,30))
        if self.set_type == 'train':
            slices = list(set(range(1,30)) - set(test_i))
        elif self.set_type == 'val':
            slices = test_i
        print('slices = ', slices)
        
        self.data_paths = []
        for i, v in enumerate(slices):
            data_path = data_root + '/' 
            self.data_paths.append({
                'imMRF': data_path + '/data/' + str(v) + '.mat',
                'Tmap': data_path + '/goals/' + str(v) + '.mat',
                'mask': data_path + '/data/' + str(v) + '.mat'
                })
