import time
from options.train_options import TrainOptions
# from data.data_loader import CreateDataLoader
# from models.models import create_model
import torch.utils.data
from util.visualizer import Visualizer
import scipy.io as sio
import util.util as util
import numpy
import argparse
import subprocess
import os
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--input_nc', type=int, default=192*2*3, help='# of input image channels')
parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels')
parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
#    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
#    parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
parser.add_argument('--which_model_netG', type=str, default='UniNet_init', help='selects model to use for netG')
#    parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
parser.add_argument('--name', type=str, default='3D', help='name of the experiment. It decides where to store samples and models')
# parser.add_argument('--dataset', type=str, default='single_dataset', help='chooses how datasets are loaded. [mrf_dataset | single_dataset]')
#    parser.add_argument('--model', type=str, default='cycle_gan', help='chooses which model to use. cycle_gan, pix2pix, test')
parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
parser.add_argument('--checkpoints_dir', type=str, default='.', help='models are saved here')
parser.add_argument('--norm', type=str, default='batch', help='instance normalization or batch normalization')
parser.add_argument('--no_dropout', default=True, action='store_true', help='no dropout for the generator')
#    parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
parser.add_argument('--goal_type', type=str, default='T1T2', help='Goal type (T1 or T2)')
parser.add_argument('--patchSize', type=int, default=64, help='patch length & width')
parser.add_argument('--PCA_n', type=int, default=17, help='number of eigen vectors in PCA')
parser.add_argument('--PCA', action='store_true', default=False, help='use PCA')
parser.add_argument('--at', action='store_true', default=False, help='Auto-context model')
parser.add_argument('--data_norm', type=str, default='energy', help='data normalization method')
parser.add_argument('--mask', type=str, default='tight', help='mask type')
parser.add_argument('--patchStride', type=int, default=32, help='patch stride')
parser.add_argument('--new_data_format', action='store_true', default=False, help='use new data format')
parser.add_argument('--PreNetwork_path', type=str, default='.', help='pretrained network\'s path')
parser.add_argument('--num_D', type=int, default=46, help='# of features')
parser.add_argument('--FNN_depth', type=int, default=4, help='depth of FNN')
parser.add_argument('--FNN_decrease', type=int, default=0, help='FNN features decrease by layer')
parser.add_argument('--Unet_struc', type=str, default='3ds', help='Unet structure')
parser.add_argument('--test_i', type=int, default=5, help='1~6, index of test subject')
parser.add_argument('--multiloss', action='store_true', default=False, help='multi-scale loss')
parser.add_argument('--multiloss_f', type=float, default=1.0, help='factor of multiloss')
parser.add_argument('--magnitude', action='store_true', default=False, help='only input magnitude')
parser.add_argument('--multi_slice_n', type=int, default=3, help='number of slices as input (for 3D data)')

parser.add_argument('--print_freq', type=int, default=1, help='frequency of showing training results on console')
parser.add_argument('--save_epoch_freq', type=int, default=50, help='frequency of saving checkpoints at the end of epochs')
parser.add_argument('--continue_train', action='store_true', default=False, help='continue training: load the latest model')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
parser.add_argument('--niter', type=int, default=40, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=10, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--lr_PCA', type=float, default=0.0, help='initial learning rate for adam for PCA parameters')
#    parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
#    parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--criterion', type=str, default='L1re', help='backloss criterion')
parser.add_argument('--augmentation', default=True, action='store_true', help='data augmentation')
parser.add_argument('--gradloss', action='store_true', default=False, help='add gradient loss')
parser.add_argument('--gan', action='store_true', default=False, help='use gan?')
parser.add_argument('--gan_lamda_G', action='store_true', default=1.0, help='weight for gan loss in G')



opt = parser.parse_args()
opt.isTrain = True
host = subprocess.check_output('hostname').decode('utf-8')[:-1]
if host == 'stilson' or host == 'andrew' or host == 'wiggin':
    opt.checkpoints_dir = '/raid/zhenghan/checkpoints'
elif host == 'badin' or host == 'bogue' or host == 'burgaw':
    opt.checkpoints_dir = '/shenlab/local/zhenghan/checkpoints'
else:
    raise ValueError("cannot decide checkpoints_dir, server '%s' not recognized." % host)
args = vars(opt)
print('------------ Options -------------')
for k, v in sorted(args.items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')
expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
util.mkdirs(expr_dir)
opt.file_name = os.path.join(expr_dir, 'log.txt')
with open(opt.file_name, 'wt') as log_file:
    log_file.write('------------ Options -------------\n')
    for k, v in sorted(args.items()):
        log_file.write('%s: %s\n' % (str(k), str(v)))
    log_file.write('-------------- End ----------------\n')


from data.threeD_dataset import MRFDataset

dataset_train = MRFDataset()
print("dataset_train [%s] was created" % (dataset_train.name()))
opt.set_type = 'train'
dataset_train.initialize(opt)
dataloader_train = torch.utils.data.DataLoader(dataset_train, 
    batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.nThreads))


dataset_test = MRFDataset()
print("dataset_test [%s] was created" % (dataset_test.name()))
opt.set_type = 'val'
dataset_test.initialize(opt)
dataloader_val = torch.utils.data.DataLoader(dataset_test, 
    batch_size=1, shuffle=False, num_workers=1)


gpu_id = util.get_vacant_gpu()
torch.cuda.set_device(gpu_id)
opt.gpu_ids=[gpu_id]
print('select gpu # %d' % gpu_id)


from models.simple_model import SimpleModel
model = SimpleModel()
model.initialize(opt)
print("model [%s] was created" % (model.name()))


visualizer = Visualizer(opt)
total_steps = 0

# util.mkdir(opt.checkpoints_dir+'/'+opt.name+'/results')

if opt.goal_type == 'T1':
    nf = 5000
elif opt.goal_type == 'T2':
    nf = 500
elif opt.goal_type == 'T1T2':
    nf = numpy.array([[[[5000],[500]]]]).astype('float32')


for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0

    dataset_train.load_data()
    for i, data in enumerate(dataloader_train):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()


        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save(epoch)
        
    if True:
        print('validation')

        fake_B_temp = numpy.zeros([256, 256, 96]).astype('float32')
        ground_B_temp = numpy.zeros([256, 256, 96]).astype('float32')
        mask_temp = numpy.zeros([256, 256, 96]).astype('float32')
        for i, data in enumerate(dataloader_val):
            model.set_input(data)
            model.test()
            errors = model.get_current_errors()          
            visualizer.print_current_errors(epoch, -(i+1), errors, 0)
            
            visual_result = model.get_current_visuals()
            # visual_result['fake_B'] = visual_result['fake_B'].transpose(3,2,1,0) * nf
            # visual_result['ground_B'] = visual_result['ground_B'].transpose(3,2,1,0) * nf
            # visual_result['mask'] = visual_result['mask'].transpose(3,2,1,0)
            # image_path = visual_result['image_path'][0]
            # sio.savemat(opt.checkpoints_dir+'/'+opt.name+'/latest_'+image_path+'.mat',{'visual_result':visual_result})
            fake_B_temp[:, :, i + int((multi_slice_n-1)/2)] = visual_result['fake_B']
            ground_B_temp[:, :, i + int((multi_slice_n-1)/2)] = visual_result['ground_B']
            mask_temp[:, :, i + int((multi_slice_n-1)/2)] = visual_result['mask']

        visual_result['fake_B'] = fake_B_temp
        visual_result['ground_B'] = ground_B_temp
        visual_result['mask'] = mask_temp
        
        sio.savemat(opt.checkpoints_dir+'/'+opt.name+'/latest.mat',{'visual_result':visual_result})
        
        print('end of validation')

    if epoch % 10 == 0:
        model.save('latest')
        print('saved the model and err at the end of epoch %d, iters %d' %
                (epoch, total_steps))
    
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()

print('saving the final model')
model.save('final')
