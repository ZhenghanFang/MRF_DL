import torch.utils.data as data
# from PIL import Image
# import torchvision.transforms as transforms
import numpy
import util.util as util
import os
import random
import torch
import h5py
import time

class BaseDataset(data.Dataset):
    def __init__(self):
        # print('__init__')
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

    def initialize_base(self, opt):
        self.opt = opt
        self.set_type = opt.set_type
        self.device = torch.device('cuda' if self.opt.gpu_ids else 'cpu')

        if opt.isTrain:
            self.augmentation = opt.augmentation
        else:
            self.augmentation = False

        self.get_paths()

        if self.set_type == 'val':
            self.load_data(self.data_paths)

        if self.set_type == 'train':
            self.patchSize = opt.patchSize
            self.switch = opt.switch
            if self.switch:
                self.current_index_i = 0
                self.load_n_eachEpoch = opt.load_n_eachEpoch
            else:
                self.load_data(self.data_paths)

    def augment(self, sample, hflip=True, rot=True):
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        def _augment(img):
            if hflip: img = img[:, :, ::-1]
            if vflip: img = img[:, ::-1, :]
            if rot90: img = img.transpose(0, 2, 1)
            
            return img

        return {k:_augment(v) for k,v in sample.items()}

    def augment_torch(self, sample, hflip=True, rot=True):
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        def _augment(img):
            if hflip: img = img.flip(2)
            if vflip: img = img.flip(1)
            if rot90: img = img.permute(0, 2, 1)
            
            return img

        return {k:_augment(v) for k,v in sample.items()}

    # def np2Tensor(self, sample):
    #     if self.opt.half:
    #         return {k:torch.from_numpy(v.astype('float32')) for k,v in sample.items()}
    #     else:
    #         return {k:torch.from_numpy(v) for k,v in sample.items()}

    def np2Tensor(self, sample):
        return {k:torch.from_numpy(v).to(self.device).float() for k,v in sample.items()}

    def np_copy(self, sample):
        return {k:v.copy() for k,v in sample.items()}

    def extractPatch(self, patch_i_1, patch_i_2, patchSize, sample):
        return {k:v[:, patch_i_1:patch_i_1+patchSize, patch_i_2:patch_i_2+patchSize] for k,v in sample.items()}

    def filter_patch_pos(self, mask, patchSize):
        imgSize = mask.shape[1]
        while True:
            patch_i_1 = random.randint(0, imgSize - patchSize)
            patch_i_2 = random.randint(0, imgSize - patchSize)
            mask_t = mask[:, patch_i_1:patch_i_1+patchSize, patch_i_2:patch_i_2+patchSize]
            if mask_t.sum() > 0.01 * mask_t.size:
                return patch_i_1, patch_i_2

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
            t = numpy.mean(A_img ** 2, axis=0) * 2
            A_img = A_img / (t[numpy.newaxis,:,:] ** 0.5) / 36
        return A_img

    def preprocess_Tmap(self, T1map, T2map):
        Tmap = numpy.stack((T1map,T2map), axis=0).transpose(2,1,0)
        Tmap = util.preprocess_tissue_property(Tmap).transpose(2,1,0)
        return Tmap

    def preprocess_mask(self, mask):
        return mask[numpy.newaxis,:,:]

    def load_data(self, data_paths):
        self.data = []
        for p in data_paths:
            self.data.append(self.load_dataset(p))

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

    def load_from_file(self, fileName, d_type):
        file = h5py.File(fileName, 'r')
        if d_type == 'imMRF':
            imMRF = self.read_imMRF(file)
            print("load imMRF")
            data = self.preprocess_imMRF(imMRF, flip=self.flipimMRF)
        elif d_type == 'Tmap':
            T1map, T2map = self.read_Tmap(file)
            data = self.preprocess_Tmap(T1map, T2map)
        elif d_type == 'mask':
            data = self.preprocess_mask(self.read_mask(file))
        else:
            raise NotImplementedError('data type [%s] is not recognized' % d_type)
        if self.opt.half:
            data = data.astype('float16')
        return data

    def read_imMRF(self, file):
        return file['imMRF'][0:self.opt.input_nc//2]

    def read_Tmap(self, file):
        return file['t1big'][:], file['t2big'][:]

    def read_mask(self, file):
        return file['mask'][:]

    def switch_data(self):
        if not self.switch:
            return
        if self.current_index_i == 0:
            self.index = list(range(len(self.data_paths)))
            random.shuffle(self.index)
        data_paths = []
        for i in range(self.load_n_eachEpoch):
            data_paths.append(self.data_paths[self.index[self.current_index_i]])
            self.current_index_i += 1
            if self.current_index_i == len(self.index):
                self.current_index_i = 0
                break
        self.load_data(data_paths)

    def __getitem__(self, index):
        dataset_i = index % len(self.data)

        if self.set_type == 'val':
            sample = {}
            sample['input_G'], sample['label_G'], sample['mask'] = (
                self.data[dataset_i]['imMRF'],
                self.data[dataset_i]['Tmap'],
                self.data[dataset_i]['mask']
                )
            sample = self.np2Tensor(sample)
        elif self.set_type == 'train':
            start = time.time()
            sample = self.get_patch(dataset_i)
            sample = self.transform_train(sample)
            # print('before aug', time.time()-start)
            sample = self.np2Tensor(sample)
            if self.augmentation:
                sample = self.augment_torch(sample)
                # print('after aug', time.time()-start)

        # sample = self.np_copy(sample)
        # print('after copy', time.time()-start)
        # sample = self.np2Tensor(sample)
        # print('after toTensor', time.time()-start)

        return {'A': sample['input_G'], 'B': sample['label_G'], 'mask': sample['mask'], 'A_paths': self.data[dataset_i]['dataset_path']}

    def transform_train(self, sample):
        return sample

    def get_patch(self, dataset_i):
        patchSize = self.patchSize
        time_start = time.time()
        patch_i_1, patch_i_2 = self.filter_patch_pos(self.data[dataset_i]['mask'], patchSize)
        sample = {}
        sample['mask'], sample['input_G'], sample['label_G'] = (
            self.data[dataset_i]['mask'],
            self.data[dataset_i]['imMRF'],
            self.data[dataset_i]['Tmap'])
        sample = self.extractPatch(patch_i_1, patch_i_2, patchSize, sample)
        print(sample.shape)
        return sample

    def __len__(self):
        if self.set_type == 'train':
            imgSize = self.data[0]['mask'].shape[1]
            return len(self.data) * int((imgSize**2)/(self.opt.patchStride**2)) * 1
        elif self.set_type == 'val':
            return len(self.data)

    def get_dataset_path(self, data_path):
        return data_path['imMRF']

'''
def get_transform(opt):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)
'''
