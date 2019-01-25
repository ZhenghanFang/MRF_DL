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
        if opt.isTrain:
            self.augmentation = opt.augmentation
        else:
            self.augmentation = False
        # self.augmentation = False
        self.augmentation_type = 'flip'
        self.goal_type = 'dict'
        self.mask_type = opt.mask

        self.get_paths(opt)
        self.A_imgs = []
        self.B_imgs = []
        self.masks = []

        for i, A_path in enumerate(self.A_paths):
            A_path1 = A_path
            f = h5py.File(A_path1)
            A_img = f['imMRF'][0:int(opt.input_nc/2), :, :]
            A_img = numpy.transpose(A_img)
            A_img = numpy.concatenate((A_img['real'],A_img['imag']), axis=2).astype('float32')
            # A_img = numpy.concatenate((A_img[:,:,0:int(opt.input_nc/2)],A_img[:,:,2304:2304+int(opt.input_nc/2)]), axis=2)
            f.close()

            # normalization
            if opt.data_norm == 'non':
                print("no normalization")
                pass
            else:
                start = time.time()
                t = numpy.mean(A_img ** 2, axis=2) * 2
                t = t[:,:,numpy.newaxis]
                A_img = A_img / (t**0.5)
                A_img = A_img / 36
                end = time.time()
                util.print_log('Time for normalizing energy: %.5fs ' % (end-start), opt.file_name)
            
            # magnitude
            if opt.magnitude:
                ntp = int(opt.input_nc/2)
                A_img_mag = numpy.copy(A_img[:,:,0:ntp])
                for k in range(ntp):
                    A_img_mag[:,:,k] = (A_img[:,:,k] ** 2 + A_img[:,:,ntp+k] ** 2) ** 0.5
                A_img = A_img_mag

            # set background signals as zero
            # f = h5py.File(self.mask_large_paths[i])
            # mask_large = numpy.transpose(f['mask']).astype('float32')
            # f.close()
            # A_img = A_img * mask_large[:,:,numpy.newaxis]

            # load mask
            if self.mask_type == 'tight':
                f = h5py.File(self.mask_tight_paths[i])
            else:
                f = h5py.File(self.mask_large_paths[i])
            mask = numpy.transpose(f['mask']).astype('float32')
            f.close()

            # clear mask
            # mask = mask * 0 + 1
            
            # load ground truth
            if self.goal_type == 'densedict':
                f = h5py.File(self.goal_densedict_paths[i])
            else:
                f = h5py.File(self.goal_paths[i])
            T1 = numpy.transpose(f['t1big']).astype('float32')
            T1 = T1[:,:,numpy.newaxis]
            T2 = numpy.transpose(f['t2big']).astype('float32')
            T2 = T2[:,:,numpy.newaxis]
            B_img = numpy.concatenate((T1,T2), axis=2)
            B_img = util.preprocess_tissue_property(B_img)
            if self.goal_type == 'densedict':
                B_img = numpy.flip(B_img, (0, 1)).copy()
            f.close()
            
            mask = mask[:,:,numpy.newaxis]
            assert A_img.ndim==3 and B_img.ndim==3, "# of dim is not 3 for training image"
            
            if opt.set_type == 'train':
                A_img = A_img[40:216,52:236,:]
                B_img = B_img[40:216,52:236,:]
                mask = mask[40:216,52:236,:]
                        
            A_img = torch.from_numpy(A_img)
            B_img = torch.from_numpy(B_img)
            mask = torch.from_numpy(mask)

            if opt.input_nc == 2304 * 2:
                print('half float16')
                A_img = A_img.half()
                B_img = B_img.half()
                mask = mask.half()

            A_img = A_img.permute(2,0,1)
            B_img = B_img.permute(2,0,1)
            mask = mask.permute(2,0,1)
            
            self.A_imgs.append(A_img)
            self.B_imgs.append(B_img)
            self.masks.append(mask)
            print("loaded image: %s" % A_path)


        self.num_imgs = len(self.A_imgs)
        if opt.patchSize != 0:
            # self.num_patch = self.num_imgs*int((self.A_imgs[0].shape[1]*self.A_imgs[0].shape[2])/(opt.patchSize**2))
            # self.num_patch = math.ceil(self.num_patch/opt.batchSize)*opt.batchSize
            self.get_patch_pos(opt)


    def __getitem__(self, index):
        start = time.time()
        index_A = index % self.num_imgs
        # A_path = self.A_paths[index_A]
        A_path = self.A_paths[index_A]
        A_img = self.A_imgs[index_A]
        B_img = self.B_imgs[index_A]
        mask = self.masks[index_A]

        if self.patchSize != 0:
            

            # random crop
            
            patch_size = self.patchSize
            
            # A_position0, A_position1 = random.randint(0,A_img.shape[1]-patch_size), random.randint(0,A_img.shape[2]-patch_size)
            A_position0, A_position1 = self.patch_pos[index // self.num_imgs % len(self.patch_pos)]

            A_position0 = min(A_position0,A_img.shape[1]-patch_size)
            A_position1 = min(A_position1,A_img.shape[2]-patch_size)

            A_img = A_img[:, A_position0:A_position0+patch_size, A_position1:A_position1+patch_size]
            B_img = B_img[:, A_position0:A_position0+patch_size, A_position1:A_position1+patch_size]
            mask = mask[:, A_position0:A_position0+patch_size, A_position1:A_position1+patch_size]

            # print('before aug', time.time()-start)
            if self.augmentation:
                # A_img, B_img, mask = self.augment(A_img, B_img, mask)
                A_img, B_img, mask = self.transform(index, A_img, B_img, mask)
                # print('after aug', time.time()-start)

        return {'A': A_img.float(), 'B': B_img.float(), 'mask': mask.float(), 'A_paths': A_path}

    def __len__(self):
        if self.patchSize == 0:
            return self.num_imgs
        else:
            return self.num_patch


    def name(self):
        return 'MRFDataset'


    def get_patch_pos(self, opt):
        pSize, pStride = opt.patchSize, opt.patchStride
        imSize = (self.A_imgs[0].shape[1], self.A_imgs[0].shape[2])
        start0, start1 = random.randint(0,pStride-1), random.randint(0,pStride-1)
        pos0 = list(range(start0,imSize[0]-pSize+1,pStride))
        pos1 = list(range(start1,imSize[1]-pSize+1,pStride))
        patch_pos = []
        for k in pos0:
            for j in pos1:
                patch_pos.append([k,j])
        # print(patch_pos)
        self.patch_pos = patch_pos
        self.num_patch = self.num_imgs*len(self.patch_pos)
        if self.augmentation:
            if self.augmentation_type == 'flip':
                self.num_patch = self.num_patch*4
            elif self.augmentation_type == 'flip+rotate':
                self.num_patch = self.num_patch*8
            else:
                raise NotImplementedError('Augmentation type [%s] is not recognized' % self.augmentation_type)
        
    
    def get_paths(self, opt):
        if opt.onMAC:
            d_root = '/Users/zhenghanfang/Desktop/standard_MRF/DataNewDictionary/'
        else:
            d_root = '/shenlab/lab_stor/zhenghan/data/MRF/DataNewDictionary/'
        person_path = ['20180206data/180114','20180206data/180124','20180206data/180131_1','20180206data/180131_2','20180206data/180202','newMRFData_041218/180408_1','newMRFData_041218/180408_2']
        slice_N = [12,12,12,12,12,10,12]
        # slice_N = [1,1,1,1,1,1,1]
        test_i = opt.test_i
        if opt.set_type == 'train':
            person = list(range(1,test_i))+list(range(test_i+1,7))
        else:
            person = list(range(test_i,test_i+1))
        # person = list(range(1,7))
            
        A_paths = []
        mask_large_paths = []
        mask_tight_paths = []
        goal_paths = []
        goal_densedict_paths = []
        for i in range(len(person)):
            a = os.listdir(d_root+person_path[person[i]-1])
            for p in a:
                if p[0] == '.':
                    a.remove(p)
            for j in range(slice_N[person[i]-1]):
                A_paths.append(d_root+person_path[person[i]-1]+'/'+a[j]+'/imMRF.mat')
                mask_large_paths.append(d_root+'mask_large/sub_'+str(person[i])+'/'+str(j+1)+'.mat')
                mask_tight_paths.append(d_root+'Data_Qian_skull_h5/'+str(person[i])+'/'+str(j+1)+'-skull.mat')
                goal_paths.append(d_root+'PatternMatching_2304/sub_'+str(person[i])+'/'+str(j+1)+'/patternmatching.mat')
                goal_densedict_paths.append(d_root+person_path[person[i]-1]+'/'+a[j]+'/patternmatching_densedict.mat')

        self.A_paths = A_paths
        self.mask_large_paths = mask_large_paths
        self.mask_tight_paths = mask_tight_paths
        self.goal_paths = goal_paths
        self.goal_densedict_paths = goal_densedict_paths

    def transform(self, index, A_img, B_img, mask):
        t = index // (self.num_imgs*len(self.patch_pos))
        if t == 1:
            A_img = torch.from_numpy(numpy.flip(A_img.numpy(),1).copy())
            B_img = torch.from_numpy(numpy.flip(B_img.numpy(),1).copy())
            mask = torch.from_numpy(numpy.flip(mask.numpy(),1).copy())
        elif t == 2:
            A_img = torch.from_numpy(numpy.flip(A_img.numpy(),2).copy())
            B_img = torch.from_numpy(numpy.flip(B_img.numpy(),2).copy())
            mask = torch.from_numpy(numpy.flip(mask.numpy(),2).copy())
        elif t == 3:
            A_img = torch.from_numpy(numpy.flip(A_img.numpy(),1).copy())
            B_img = torch.from_numpy(numpy.flip(B_img.numpy(),1).copy())
            mask = torch.from_numpy(numpy.flip(mask.numpy(),1).copy())
            A_img = torch.from_numpy(numpy.flip(A_img.numpy(),2).copy())
            B_img = torch.from_numpy(numpy.flip(B_img.numpy(),2).copy())
            mask = torch.from_numpy(numpy.flip(mask.numpy(),2).copy())
        elif t == 4 or t == 5 or t == 6 or t == 7:
            A_img = A_img.numpy().copy().transpose(0,2,1)
            B_img = B_img.numpy().copy().transpose(0,2,1)
            mask = mask.numpy().copy().transpose(0,2,1)
            if t == 4:
                A_img = torch.from_numpy(A_img.copy())
                B_img = torch.from_numpy(B_img.copy())
                mask = torch.from_numpy(mask.copy())
            elif t == 5:
                A_img = torch.from_numpy(numpy.flip(A_img,1).copy())
                B_img = torch.from_numpy(numpy.flip(B_img,1).copy())
                mask = torch.from_numpy(numpy.flip(mask,1).copy())
            elif t == 6:
                A_img = torch.from_numpy(numpy.flip(A_img,2).copy())
                B_img = torch.from_numpy(numpy.flip(B_img,2).copy())
                mask = torch.from_numpy(numpy.flip(mask,2).copy())
            elif t == 7:
                A_img = torch.from_numpy(numpy.flip(A_img,1).copy())
                B_img = torch.from_numpy(numpy.flip(B_img,1).copy())
                mask = torch.from_numpy(numpy.flip(mask,1).copy())
                A_img = torch.from_numpy(numpy.flip(A_img,2).copy())
                B_img = torch.from_numpy(numpy.flip(B_img,2).copy())
                mask = torch.from_numpy(numpy.flip(mask,2).copy())

        return A_img, B_img, mask
    
