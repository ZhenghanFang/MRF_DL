# import os.path
# import torchvision.transforms as transforms
# from data.base_dataset import BaseDataset, get_transform
from data.highres_dataset import MRFDataset as highresDataset
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

class MRFDataset(highresDataset):
    def initialize(self, opt):
        self.flipimMRF = False
        self.initialize_base(opt)

    def name(self):
        return 'single_Dataset_2'
    
    def load_dataset(self, data_path):
        print('load dataset: ', data_path)

        data = {}
        for k,v in data_path.items():
            data[k] = self.load_from_file(v, k)
            
        if self.opt.zerobg:
            data['imMRF'] = data['imMRF'] * data['mask']

        # dataset_path = os.path.splitext(os.path.split(imMRF_path)[-1])[0]
        data['dataset_path'] = self.get_dataset_path(data_path)
        return data


    def get_paths(self):
        if self.opt.onMAC:
            d_root = '/Users/zhenghanfang/Desktop/standard_MRF/DataNewDictionary/'
        else:
            d_root = '/shenlab/lab_stor/zhenghan/data/MRF/DataNewDictionary/'
        person_path = ['20180206data/180114','20180206data/180124','20180206data/180131_1','20180206data/180131_2','20180206data/180202','newMRFData_041218/180408_1','newMRFData_041218/180408_2']
        slice_N = [12,12,12,12,12,10,12]
        # slice_N = [1,1,1,1,1,1,1]
        test_i = self.opt.test_i
        if self.opt.set_type == 'train':
            person = list(range(1,test_i))+list(range(test_i+1,7))
        else:
            person = list(range(test_i,test_i+1))
        # person = list(range(1,7))

        self.data_paths = []
        for i in range(len(person)):
            a = os.listdir(d_root+person_path[person[i]-1])
            for p in a:
                if p[0] == '.':
                    a.remove(p)
            for j in range(slice_N[person[i]-1]):
                self.data_paths.append({
                    'imMRF': d_root+person_path[person[i]-1]+'/'+a[j]+'/imMRF.mat',
                    'Tmap': d_root+'PatternMatching_2304/sub_'+str(person[i])+'/'+str(j+1)+'/patternmatching.mat', # sparse dict
                    # 'Tmap': d_root+person_path[person[i]-1]+'/'+a[j]+'/patternmatching_densedict.mat', # dense dict
                    'mask': d_root+'mask_large/sub_'+str(person[i])+'/'+str(j+1)+'.mat' # large mask
                    # 'mask': d_root+'Data_Qian_skull_h5/'+str(person[i])+'/'+str(j+1)+'-skull.mat' # small mask
                    })
