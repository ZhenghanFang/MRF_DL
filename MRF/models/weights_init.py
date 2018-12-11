import numpy as np
import h5py
import torch

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def weights_init_PCA(net, opt):
    V_path = '/shenlab/lab_stor/zhenghan/data/MRF/V.mat'
    f = h5py.File(V_path)
    V = np.transpose(f['V'])
    f.close()
    V = V[0:int(opt.input_nc/2),0:opt.PCA_n]
    V1 = np.concatenate((V['real'], V['imag']), axis=0)
    V2 = np.concatenate((-V['imag'], V['real']), axis=0)
    V = np.concatenate((V1, V2), axis=1)
    V = V.transpose()
    V = V[:, :, np.newaxis, np.newaxis]
    V = torch.from_numpy(V.astype('float32'))
    
    m = net.model[0]
    classname = m.__class__.__name__
    m.weight.data.copy_(V)
    # print(m.weight.data)
    if hasattr(m.bias, 'data'):
        raise ValueError('PCA layer has bias')