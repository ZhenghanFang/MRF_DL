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
        file = h5py.File(fileName, 'r')
        if d_type == 'imMRF':
            data = file['visual_result']['fake_B'][:] / numpy.array([[[5000], [500]]])
        elif d_type == 'Tmap':
            T1map, T2map = self.read_Tmap(file)
            data = file['visual_result']['ground_B'][:] / numpy.array([[[5000], [500]]]) - file['visual_result']['fake_B'][:] / numpy.array([[[5000], [500]]])
        elif d_type == 'mask':
            data = file['visual_result']['mask'][:]
        else:
            raise NotImplementedError('data type [%s] is not recognized' % d_type)
        if self.opt.half:
            data = data.astype('float16')
        return data

    def get_paths(self):
        if self.opt.onMAC:
            d_root = '/Users/zhenghanfang/Desktop/standard_MRF/DataNewDictionary/'
        else:
            d_root = '/shenlab/lab_stor/zhenghan/data/MRF/Uni_46_T288/'
        person_path = ['Uni_46_T288_sub1_test','Uni_46_T288_sub2_test','Uni_46_T288_sub3_test','Uni_46_T288_sub4_test','Uni_46_T288_sub5_test','Uni_46_T288_sub6_test']
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
