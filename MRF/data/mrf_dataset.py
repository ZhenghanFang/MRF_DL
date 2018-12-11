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
import skimage.transform
import time
import scipy.io as sio

class MRFDataset(BaseDataset):
    def initialize(self, opt):

        self.augmentation = opt.augmentation

        self.at = opt.at
        
        self.get_paths(opt)
        self.A_imgs = []
        self.B_imgs = []
        self.masks = []

        for i, A_path in enumerate(self.A_paths):
            # A_path1 = '/raid/zhenghan/data/MRF/data/'+A_path+'.mat'
            # A_path1 = '/shenlab/lab_stor/zhenghan/data/MRF/data_original/'+A_path+'.mat'
            A_path1 = A_path
            f = h5py.File(A_path1)
            A_img = numpy.transpose(f['imSVD'])
            if not opt.new_data_format:
                A_img = numpy.concatenate((A_img['real'],A_img['imag']), axis=2).astype('float32')
                A_img = numpy.concatenate((A_img[:,:,0:int(opt.input_nc/2)],A_img[:,:,2304:2304+int(opt.input_nc/2)]), axis=2)
            else:
                A_img = numpy.concatenate((A_img[:,:,0:int(opt.input_nc/2)],A_img[:,:,int(A_img.shape[2]/2):int(A_img.shape[2]/2)+int(opt.input_nc/2)]), axis=2)
            f.close()

            if opt.data_norm == 'minmax':
                A_img = (A_img-A_img.min())/(A_img.max()-A_img.min())*2-1
            elif opt.data_norm == 'energy':
                t = numpy.mean(A_img ** 2, axis=2) * 2
                t = t[:,:,numpy.newaxis]
                A_img = A_img / (t**0.5)
                A_img = A_img / 36
            elif opt.data_norm == 'non':
                pass
            else:
                raise ValueError('data normalization method %s not recognized' % opt.data_norm)
            
            #B_img = numpy.transpose(f['goal']).astype('float32')
            if opt.mask == 'loose':
                f = h5py.File(A_path1)
            elif opt.mask == 'tight':
                # f = h5py.File('/shenlab/lab_stor/zhenghan/data/MRF/mask_tight/'+A_path+'.mat')
                f = h5py.File(self.mask_paths[i])
            else:
                raise ValueError('mask type %s not recognized' % opt.mask)
            mask = numpy.transpose(f['mask']).astype('float32')
            f.close()
            

            f = h5py.File(self.goal_paths[i])
            T1 = numpy.transpose(f['t1big']).astype('float32')
            T1 = T1/5000
            T1 = T1[:,:,numpy.newaxis]
            T2 = numpy.transpose(f['t2big']).astype('float32')
            T2 = T2/500
            T2 = T2[:,:,numpy.newaxis]

            if opt.goal_type == 'T1':
                B_img = T1
            elif opt.goal_type == 'T2':
                B_img = T2
            elif opt.goal_type == 'T1T2':
                B_img = numpy.concatenate((T1,T2), axis=2)
            f.close()
            
            mask = mask[:,:,numpy.newaxis]
            #if B_img.ndim==2:
            #    B_img = B_img[:,:,numpy.newaxis]
            assert A_img.ndim==3 and B_img.ndim==3, "# of dim is not 3 for training image"
            
            '''
            A_img = A_img[53:201,58:229,:]
            B_img = B_img[53:201,58:229,:]
            mask = mask[53:201,58:229,:]
            '''
            A_img = A_img[40:216,52:236,:]
            B_img = B_img[40:216,52:236,:]
            mask = mask[40:216,52:236,:]
                        
            A_img = torch.from_numpy(A_img)
            B_img = torch.from_numpy(B_img)
            mask = torch.from_numpy(mask)
            
            
            #if opt.data_GPU:
            #    A_img = A_img.cuda()
            #    B_img = B_img.cuda()
            #    mask = mask.cuda()

            A_img = A_img.permute(2,0,1)
            B_img = B_img.permute(2,0,1)
            mask = mask.permute(2,0,1)

            if self.at:
                prev_path = '/shenlab/lab_stor/zhenghan/data/MRF/at/MRFT1_2304_cnn_small_newT1/'+A_path+'.mat'
                f = h5py.File(prev_path)
                vr = (f['visual_result'])
                prev = numpy.transpose(vr['fake_B']).astype('float32')
                f.close()
                if opt.goal_type == 'T1':
                    prev = prev/5000
                if opt.goal_type == 'T2':
                    prev = prev/500
                prev = prev[:,:,numpy.newaxis]
                prev = torch.from_numpy(prev)
                prev = prev.permute(2,0,1)
                A_img = torch.cat((A_img, prev), 0)
                # print(A_img[-1,:,:]-B_img)
            
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
        
        index_A = index % self.num_imgs
        # A_path = self.A_paths[index_A]
        A_path = ''
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

            if self.augmentation:
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

        
        return {'A': A_img, 'B': B_img, 'mask': mask, 'A_paths': A_path}

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
            self.num_patch = self.num_patch*4

    
    def get_paths(self, opt):
        d_root = '/shenlab/lab_stor/zhenghan/data/MRF/DataNewDictionary/'
        slice_N = [12,12,12,12,12,10,12]
        test_i = 5
        if opt.set_type == 'train':
            person = list(range(1,test_i))+list(range(test_i+1,7))
        else:
            person = list(range(test_i,test_i+1))
            
        A_paths = []
        mask_paths = []
        goal_paths = []
        for i in range(len(person)):
            for j in range(slice_N[person[i]-1]):
                A_paths.append(d_root+'imSVD_576_100/sub_'+str(person[i])+'/'+str(j+1)+'/imSVD.mat')
                mask_paths.append(d_root+'mask_large/sub_'+str(person[i])+'/'+str(j+1)+'.mat')
                goal_paths.append(d_root+'PatternMatching_2304/sub_'+str(person[i])+'/'+str(j+1)+'/patternmatching.mat')


        self.A_paths = A_paths
        self.mask_paths = mask_paths
        self.goal_paths = goal_paths

