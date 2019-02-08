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
        return 'simulated_Dataset_noise'

    def read_mask(self, file):
        return file['immask'][:]
    
    def preprocess_imMRF(self, imMRF, flip=True):
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
            A_img = A_img * 1e-5
            t = numpy.mean(A_img ** 2, axis=0) * 2
            A_img = A_img / (t[numpy.newaxis,:,:] ** 0.5) / 36
        return A_img

    def get_paths(self):
        if self.opt.onMAC:
            d_root = '/Users/zhenghanfang/Desktop/standard_MRF/DataNewDictionary/Data_20190130/SimulatedTrainingData_Noise5/'
        else:
            d_root = '/shenlab/lab_stor/zhenghan/data/MRF/simulate/SimulatedTrainingData_Noise5/'
        person_path = ['181007', '181012-1', '181012-2', '181014-1', '181014-2']
        slice_path = [str(i) for i in range(155,174,2)]
        # slice_N = [10, 10, 10, 10, 10]
        # slice_N = [1,1,1,1,1,1]
        test_i = 0
        if self.set_type == 'train':
            slice = list(range(0,test_i)) + list(range(test_i+1,10))
        elif self.set_type == 'val':
            slice = list(range(test_i,test_i+1))
            
        self.data_paths = []
        for j in range(len(slice)):
            d_path = d_root + slice_path[j] + '/'
            self.data_paths.append({
                'imMRF': d_path + 'simu_imMRF_Noise_5.mat',
                'Tmap': d_path + 'simu_patternmatching_Noise_5.mat',
                'mask': d_path + 'simu_immask_Noise_5.mat'
                })
