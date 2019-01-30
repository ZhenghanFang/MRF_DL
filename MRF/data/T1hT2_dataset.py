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
        return 'T1hT2_Dataset'

    def load_from_file(self, fileName, d_type):
        file = sio.loadmat(fileName)
        norm_factor = numpy.array([[[5000]],[[500]]])
        if d_type == 'imMRF':
            data = file['visual_result']['fake_B'][0,0].transpose(2,1,0) / norm_factor
        elif d_type == 'Tmap':
            self.predict_error = self.opt.T1hT2_predict_error
            if self.predict_error:
                data = file['visual_result']['ground_B'][0,0].transpose(2,1,0) / norm_factor - file['visual_result']['fake_B'][0,0].transpose(2,1,0) / norm_factor
            else:
                data = file['visual_result']['ground_B'][0,0].transpose(2,1,0) / norm_factor
        elif d_type == 'mask':
            data = file['visual_result']['mask'][0,0].transpose(2,1,0)
        else:
            raise NotImplementedError('data type [%s] is not recognized' % d_type)
        if self.opt.half:
            data = data.astype('float16')
        return data

    def get_paths(self):
        if self.opt.onMAC:
            d_root = '/Users/zhenghanfang/Desktop/standard_MRF/DataNewDictionary/'
        else:
            d_root = '/shenlab/lab_stor/zhenghan/data/MRF/'
        d_root = d_root + self.opt.T1hT2_dataroot + '/'
        person_path = []
        for k in range(1,7):
            person_path.append(self.opt.T1hT2_dataroot + '_sub' + str(k) + '_test')
        # person_path = ['Uni_46_T288_sub1_test','Uni_46_T288_sub2_test','Uni_46_T288_sub3_test','Uni_46_T288_sub4_test','Uni_46_T288_sub5_test','Uni_46_T288_sub6_test']
        # person_path = ['rcab_zerobg_ar8_sub1_test','rcab_zerobg_ar8_sub2_test','rcab_zerobg_ar8_sub3_test','rcab_zerobg_ar8_sub4_test','rcab_zerobg_ar8_sub5_test','rcab_zerobg_ar8_sub6_test']
        slice_N = [12,12,12,12,12,10]
        # slice_N = [1,1,1,1,1,1]
        test_i = self.opt.test_i
        if self.opt.set_type == 'train':
            person = list(range(1,test_i))+list(range(test_i+1,7))
        else:
            person = list(range(test_i,test_i+1))
        # person = list(range(1,7))

        self.data_paths = []
        for i in range(len(person)):
            for j in range(slice_N[person[i]-1]):
                self.data_paths.append({
                    'imMRF': d_root+person_path[person[i]-1]+'/latest_'+str(j+1)+'.mat',
                    'Tmap': d_root+person_path[person[i]-1]+'/latest_'+str(j+1)+'.mat', # sparse dict
                    'mask': d_root+person_path[person[i]-1]+'/latest_'+str(j+1)+'.mat' # large mask
                    })
