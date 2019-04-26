import time
import os
from options.test_options import TestOptions
# from data.data_loader import CreateDataLoader
# from models.models import create_model
import torch.utils.data
from util.visualizer import Visualizer
# from util import html
import scipy.io as sio
import numpy
import argparse
import util.util as util
from data import getDataset
from models import getModel



parser = argparse.ArgumentParser()
parser.add_argument('--T1hT2_predict_error', action='store_true', default=False, help='predict error or T2 map?')
parser.add_argument('--T1hT2_dataroot', type=str, default='', help='data root for T1hT2')
parser.add_argument('--zerobg', action='store_true', default=False, help='set background signal as zero?')
parser.add_argument('--saved_model_path', type=str, default='/Users/zhenghanfang/raid/zhenghan/checkpoints/MRF_simu/simu_ar4/final_net_G_A.pth', help='path of model')
parser.add_argument('--data_name', type=str, default='simudata', help='name of the dataset.')
parser.add_argument('--input_nc', type=int, default=int(2304/4*2), help='# of input image channels')
parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels')
parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
#    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
#    parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
parser.add_argument('--model', type=str, default='SimpleModel', help='selects main model')
parser.add_argument('--which_model_netG', type=str, default='UniNet_init', help='selects model to use for netG')
parser.add_argument('--Unet_struc', type=str, default='3ds', help='Unet structure')
parser.add_argument('--FNN_depth', type=int, default=4, help='depth of FNN')
parser.add_argument('--num_D', type=int, default=64, help='# of features')
parser.add_argument('--FNN_decrease', type=int, default=0, help='FNN features decrease by layer')
parser.add_argument('--criterion', type=str, default='L1re', help='backloss criterion')
#    parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
#    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--dataset', type=str, default='single_dataset', help='chooses how datasets are loaded. [mrf_dataset | single_dataset]')
parser.add_argument('--half', action='store_true', default=True, help='Half precision data (float16)?')
#    parser.add_argument('--model', type=str, default='cycle_gan', help='chooses which model to use. cycle_gan, pix2pix, test')
# parser.add_argument('--checkpoints_dir', type=str, default='.', help='models are saved here')
parser.add_argument('--norm', type=str, default='batch', help='instance normalization or batch normalization')
parser.add_argument('--no_dropout', default=True, action='store_true', help='no dropout for the generator')
#    parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
parser.add_argument('--goal_type', type=str, default='T1T2', help='Goal type (T1 or T2)')
parser.add_argument('--patchSize', type=int, default=0, help='patch length & width')
parser.add_argument('--PCA_n', type=int, default=17, help='number of eigen vectors in PCA')
parser.add_argument('--PCA', action='store_true', default=False, help='use PCA')
parser.add_argument('--at', action='store_true', default=False, help='Auto-context model')
parser.add_argument('--data_norm', type=str, default='energy', help='data normalization method')
parser.add_argument('--mask', type=str, default='tight', help='mask type')
# parser.add_argument('--patchStride', type=int, default=32, help='patch stride')
parser.add_argument('--new_data_format', action='store_true', default=False, help='use new data format')
parser.add_argument('--PreNetwork_path', type=str, default=None, help='pretrained network\'s path')
parser.add_argument('--test_i', type=int, default=5, help='1~6, index of test subject')
parser.add_argument('--onMAC', action='store_true', default=True, help='Run on iMAC')
parser.add_argument('--multiloss', action='store_true', default=False, help='multi-scale loss')
parser.add_argument('--multiloss_f', type=float, default=1.0, help='factor of multiloss')
parser.add_argument('--magnitude', action='store_true', default=False, help='only input magnitude')
parser.add_argument('--gradloss', action='store_true', default=False, help='add gradient loss')

parser.add_argument('--gan', action='store_true', default=False, help='use gan?')
parser.add_argument('--multi_slice_n', type=int, default=3, help='number of slices as input (for 3D data)')


opt = parser.parse_args()
opt.isTrain = False
opt.model_name = os.path.basename(os.path.dirname(opt.saved_model_path))
opt.results_dir = os.path.join(os.path.dirname(opt.saved_model_path), opt.model_name + '_' + opt.data_name + '_test')
args = vars(opt)
print('------------ Options -------------')
for k, v in sorted(args.items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')
util.mkdirs(opt.results_dir)
opt.file_name = os.path.join(opt.results_dir, 'log.txt')
with open(opt.file_name, 'wt') as log_file:
    log_file.write('------------ Options -------------\n')
    for k, v in sorted(args.items()):
        log_file.write('%s: %s\n' % (str(k), str(v)))
    log_file.write('-------------- End ----------------\n')



opt.gpu_ids=[]

Model = getModel(opt)
model = Model()
model.initialize(opt)
print("model [%s] was created" % (model.name()))

MRFDataset = getDataset(opt)

opt.set_type = 'val'
dataset = MRFDataset()
dataset.initialize(opt)
dataloader_test = torch.utils.data.DataLoader(dataset,
    batch_size=1, shuffle=False, num_workers=1)
dataloader_test.dataset.patchSize = 0
print("dataset [%s] was created" % (dataset.name()))

visualizer = Visualizer(opt)

if opt.goal_type == 'T1':
    nf = 5000
elif opt.goal_type == 'T2':
    nf = 500
elif opt.goal_type == 'T1T2':
    nf = numpy.array([[[[5000],[500]]]]).astype('float32')

for i, data in enumerate(dataloader_test):
    start = time.time()
    model.set_input(data)
    model.test()
    end = time.time()
    
    errors = model.get_current_errors()
    message = '(epoch: , image: %d) ' % (i+1)
    for k, v in errors.items():
        message += '%s: %.5f ' % (k, v)
    message += ' time: %.5fs' % (end-start)
    util.print_log(message, opt.file_name)
    visual_result = model.get_current_visuals()
    sio.savemat(os.path.join(opt.results_dir, 'latest_' + str(i+143) + '.mat'),{'visual_result':visual_result})

print('Test done')

