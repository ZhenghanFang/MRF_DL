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
        '''
        with open(self.A_paths_file, 'r') as f:
            self.A_paths = f.read()
        self.A_paths = self.A_paths.replace('\n',' ').split()
        '''
        
        self.get_paths(opt)
        self.A_imgs = []
        self.B_imgs = []
        self.masks = []

        for i, A_path in enumerate(self.A_paths):
            # A_path1 = '/raid/zhenghan/data/MRF/data/'+A_path+'.mat'
            # A_path1 = '/shenlab/lab_stor/zhenghan/data/MRF/data_original/'+A_path+'.mat'
            A_path1 = A_path
            f = h5py.File(A_path1)
            A_img = numpy.transpose(f['imMRF'])
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
            

            #A_path1 = '/shenlab/lab_stor/zhenghan/data/MRF/data/'+A_path+'.mat'
            #f = h5py.File(A_path1)
            #A_img2 = numpy.transpose(f['imMRF']).astype('float32')
            #f.close()
            #print(abs(A_img2-A_img).sum())
            
            '''
            if opt.goal_type == 'T2':
                A_path2='/raid/zhenghan/data/MRF/dataT2/'+A_path+'_T2.mat'
                #print(A_path2)
                f = h5py.File(A_path2)
                B_img = numpy.transpose(f['t2big']).astype('float32')
                maskT2 = numpy.transpose(f['maskT2']).astype('float32')
                
                f.close()
                mask = mask*maskT2
                
                
            if opt.goal_type == 'T1':
                A_path2='/raid/zhenghan/data/MRF/dataT1/'+A_path+'_T1.mat'
                
                f = h5py.File(A_path2)
                B_img = numpy.transpose(f['t1big']).astype('float32')
                maskT1 = numpy.transpose(f['maskT1']).astype('float32')
                
                f.close()
                mask = mask*maskT1
            '''
                
            # f = h5py.File('/raid/zhenghan/data/MRF/goals/'+A_path+'.mat')
            # f = h5py.File('/shenlab/lab_stor/zhenghan/data/MRF/goals/'+A_path+'.mat')
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


            if False:
                
                A_img = A_img.permute(1,2,0)
                B_img = B_img.permute(1,2,0)
                mask = mask.permute(1,2,0)
                
                A_img = A_img.numpy()
                B_img = B_img.numpy()
                mask = mask.numpy()

                '''
                n_scale = random.random()/5-0.1+1
                A_img = skimage.transform.rescale(A_img,n_scale)
                B_img = skimage.transform.rescale(B_img,n_scale)
                mask = skimage.transform.rescale(mask,n_scale)
                '''

                n_rotate = -10
                A_img1 = skimage.transform.rotate(A_img,n_rotate,mode='reflect')
                B_img1 = skimage.transform.rotate(B_img,n_rotate,mode='reflect')
                mask1 = skimage.transform.rotate(mask,n_rotate,mode='constant')
                
                A_img1 = torch.from_numpy(A_img1.astype('float32'))
                B_img1 = torch.from_numpy(B_img1.astype('float32'))
                mask1 = torch.from_numpy(mask1.astype('float32'))
                
                A_img1 = A_img1.permute(2,0,1)
                B_img1 = B_img1.permute(2,0,1)
                mask1 = mask1.permute(2,0,1)

                self.A_imgs.append(A_img1)
                self.B_imgs.append(B_img1)
                self.masks.append(mask1)

                n_rotate = 10
                A_img1 = skimage.transform.rotate(A_img,n_rotate,mode='reflect')
                B_img1 = skimage.transform.rotate(B_img,n_rotate,mode='reflect')
                mask1 = skimage.transform.rotate(mask,n_rotate,mode='constant')
                # sio.savemat(opt.checkpoints_dir+'/'+opt.name+'/aug_'+'.mat',{'A_img1':A_img1,'B_img1':B_img1,'mask1':mask1})


                A_img1 = torch.from_numpy(A_img1.astype('float32'))
                B_img1 = torch.from_numpy(B_img1.astype('float32'))
                mask1 = torch.from_numpy(mask1.astype('float32'))
                
                A_img1 = A_img1.permute(2,0,1)
                B_img1 = B_img1.permute(2,0,1)
                mask1 = mask1.permute(2,0,1)

                self.A_imgs.append(A_img1)
                self.B_imgs.append(B_img1)
                self.masks.append(mask1)


            
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
            self.num_patch = self.num_patch*3

    
    def get_paths(self, opt):
        d_root = '/shenlab/lab_stor/zhenghan/data/MRF/20180206data/'
        person = ['180114','180124','180131_1','180131_2','180202']
        slice_start = [181,44,43,87,154]
        test_i = 4
        if opt.set_type == 'train':
            person = person[:test_i]+person[test_i+1:]
            slice_start = slice_start[:test_i]+slice_start[test_i+1:]
        else:
            person = person[test_i:test_i+1]
            slice_start = slice_start[test_i:test_i+1]
            
        A_paths = []
        mask_paths = []
        goal_paths = []
        for i in range(len(person)):
            for j in range(12):
                A_paths.append(d_root+person[i]+'/'+str(slice_start[i]+j*2)+'/imMRF_576_47.mat')
                mask_paths.append(d_root+person[i]+'/'+str(slice_start[i]+j*2)+'/mask_tight.mat')
                goal_paths.append(d_root+person[i]+'/'+str(slice_start[i]+j*2)+'/patternmatching_2304.mat')
        

        if opt.set_type == 'train':
            d_root = '/shenlab/lab_stor/zhenghan/data/MRF/'
            for k in [1,3,5,14,15,16,18,20,23,8,9,10,11,12,13,19,21]:
                A_paths.append(d_root+'data_original_576_47/'+str(k)+'.mat')
                mask_paths.append(d_root+'mask_tight/'+str(k)+'.mat')
                goal_paths.append(d_root+'goals/'+str(k)+'.mat')
        else:
            d_root = '/shenlab/lab_stor/zhenghan/data/MRF/'
            for k in [2,17,22]:
                A_paths.append(d_root+'data_original_576_47/'+str(k)+'.mat')
                mask_paths.append(d_root+'mask_tight/'+str(k)+'.mat')
                goal_paths.append(d_root+'goals/'+str(k)+'.mat')

        self.A_paths = A_paths
        self.mask_paths = mask_paths
        self.goal_paths = goal_paths
    '''
    def get_paths(self, opt):
        A_paths = []
        mask_paths = []
        goal_paths = []        

        if opt.set_type == 'train':
            d_root = '/shenlab/lab_stor/zhenghan/data/MRF/'
            for k in [1,3,5,14,15,16,18,20,23,8,9,10,11,12,13,19,21]:
                A_paths.append(d_root+'data_original/'+str(k)+'.mat')
                mask_paths.append(d_root+'mask_tight/'+str(k)+'.mat')
                goal_paths.append(d_root+'goals/'+str(k)+'.mat')
        else:
            d_root = '/shenlab/lab_stor/zhenghan/data/MRF/'
            for k in [2,17,22]:
                A_paths.append(d_root+'data_original/'+str(k)+'.mat')
                mask_paths.append(d_root+'mask_tight/'+str(k)+'.mat')
                goal_paths.append(d_root+'goals/'+str(k)+'.mat')

        self.A_paths = A_paths
        self.mask_paths = mask_paths
        self.goal_paths = goal_paths
    '''