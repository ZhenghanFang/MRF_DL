import torch
import torch.nn as nn
import functools
from .EDSR_models.rcan import RCAN, RCAB
from .block import *
from .weights_init import *


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class SimpleCNN(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(SimpleCNN, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        
        model = []

        model += [
                  nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1,
                            bias=use_bias),
                  norm_layer(ngf),
                  nn.ReLU(True)]
        
        model += [nn.Conv2d(ngf, ngf, kernel_size=3,
                                padding=1, bias=use_bias),
                  norm_layer(ngf),
                  nn.ReLU(True)]

        model += [nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

class SimpleCNN_larger(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(SimpleCNN_larger, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        
        model = []
        model += [
                  nn.Conv2d(input_nc, 4096, kernel_size=1, padding=0,
                            bias=use_bias),
                  norm_layer(4096),
                  nn.ReLU(True)]

        model += [
                  nn.Conv2d(4096, 1024, kernel_size=3, padding=1,
                            bias=use_bias),
                  norm_layer(1024),
                  nn.ReLU(True)]
        
        model += [nn.Conv2d(1024, 512, kernel_size=3,
                                padding=1, bias=use_bias),
                  norm_layer(512),
                  nn.ReLU(True)]

        model += [nn.Conv2d(512, output_nc, kernel_size=3, padding=1)]
        #model += [nn.Tanh()]

        self.model = nn.Sequential(*model)
    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

class SimpleCNN_large(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(SimpleCNN_large, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        
        model = []

        model += [
                  nn.Conv2d(input_nc, 512, kernel_size=3, padding=1,
                            bias=use_bias),
                  norm_layer(512),
                  nn.ReLU(True)]
        
        model += [nn.Conv2d(512, 256, kernel_size=3,
                                padding=1, bias=use_bias),
                  norm_layer(256),
                  nn.ReLU(True)]

        model += [nn.Conv2d(256, output_nc, kernel_size=3, padding=1)]
        #model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

class _SimpleCNN_small(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(_SimpleCNN_small, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        
        model = []

        model += [
                  nn.Conv2d(input_nc, 1024, kernel_size=1, padding=0, bias=use_bias),
                  norm_layer(1024),
                  nn.ReLU(True)]
        
        model += [nn.Conv2d(1024, 256, kernel_size=1, padding=0, bias=use_bias),
                  norm_layer(256),
                  nn.ReLU(True)]

        model += [nn.Conv2d(256, output_nc, kernel_size=3, padding=1)]
        
        
        
        #model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class SimpleCNN_small(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(SimpleCNN_small, self).__init__()
        self.T1 = _SimpleCNN_small(opt, input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
        self.T2 = _SimpleCNN_small(opt, input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    def forward(self, input):
        return torch.cat([self.T1(input), self.T2(input)], 1)

class SimpleCNN_small_at(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(SimpleCNN_small_at, self).__init__()
        self.gpu_ids = gpu_ids
        self.model = SimpleCNN_small_at_struc(opt, input_nc, output_nc, ngf, norm_layer, use_dropout, gpu_ids)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

class SimpleCNN_small_at_struc(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(SimpleCNN_small_at_struc, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d


        model = []

        model += [
                  nn.Conv2d(input_nc, 1024, kernel_size=1, padding=0, bias=use_bias),
                  norm_layer(1024),
                  nn.ReLU(True)]

        model += [nn.Conv2d(1024, 256, kernel_size=1, padding=0, bias=use_bias),
                  norm_layer(256),
                  nn.ReLU(True)]

        model += [nn.Conv2d(256, output_nc, kernel_size=3, padding=1)]



        #model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input[:,:-1,:,:])/100 + input[:,-1,:,:]
        # return self.model(input[:,:-1,:,:])/10000000000

class SimpleCNN_small_PCA(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(SimpleCNN_small_PCA, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        PCA_n = opt.PCA_n
        
        model = []
        
        model += [nn.Conv2d(input_nc, PCA_n*2, kernel_size=1, padding=0, bias=False)]

        model += [
                  nn.Conv2d(PCA_n*2, 1024, kernel_size=1, padding=0, bias=use_bias),
                  norm_layer(1024),
                  nn.ReLU(True)]
        
        model += [nn.Conv2d(1024, 256, kernel_size=1, padding=0, bias=use_bias),
                  norm_layer(256),
                  nn.ReLU(True)]

        model += [nn.Conv2d(256, output_nc, kernel_size=3, padding=1)]
        
        
        
        #model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            # a=nn.parallel.data_parallel(self.model[0], input, self.gpu_ids)
            # print(a.data.shape)
            # print(a.data[0,2,10,10])
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

class SimpleNN(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(SimpleNN, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        #self.ngf = ngf
        self.gpu_ids = gpu_ids
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        
        model = []

        model += [
                  nn.Conv2d(input_nc, 1024, kernel_size=1, padding=0, bias=use_bias),
                  norm_layer(1024),
                  nn.ReLU(True)]
        
        model += [nn.Conv2d(1024, 512, kernel_size=1, padding=0, bias=use_bias),
                  norm_layer(512),
                  nn.ReLU(True)]
        model += [nn.Conv2d(512, 256, kernel_size=1, padding=0, bias=use_bias),
                  norm_layer(256),
                  nn.ReLU(True)]

        model += [nn.Conv2d(256, output_nc, kernel_size=1, padding=0)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

class SimpleNN_large(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(SimpleNN_large, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        #self.ngf = ngf
        self.gpu_ids = gpu_ids
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        
        model = []

        model += [
                  nn.Conv2d(input_nc, 2048, kernel_size=1, padding=0, bias=use_bias),
                  norm_layer(2048),
                  nn.ReLU(True)]
        
        model += [nn.Conv2d(2048, 1024, kernel_size=1, padding=0, bias=use_bias),
                  norm_layer(1024),
                  nn.ReLU(True)]
        model += [nn.Conv2d(1024, 512, kernel_size=1, padding=0, bias=use_bias),
                  norm_layer(512),
                  nn.ReLU(True)]

        model += [nn.Conv2d(512, output_nc, kernel_size=1, padding=0)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

class Unet_struc(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet_struc, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        model = {}

        model['b'] = [
            nn.MaxPool2d(2), 
            nn.Conv2d(self.ngf*2, self.ngf*4, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*4),
            nn.ReLU(True),
            nn.Conv2d(self.ngf*4, self.ngf*4, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf*4, self.ngf*2, kernel_size=2, stride=2)
            # norm_layer(self.ngf*2),
            # nn.ReLU(True)
            ]
        self.b = nn.Sequential(*model['b'])

        model['down2'] = [
            nn.MaxPool2d(2), 
            nn.Conv2d(self.ngf, self.ngf*2, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*2),
            nn.ReLU(True),
            nn.Conv2d(self.ngf*2, self.ngf*2, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*2),
            nn.ReLU(True)
            ]
        self.down2 = nn.Sequential(*model['down2'])

        model['down1'] = [
            nn.Conv2d(self.input_nc, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True)
            ]
        self.down1 = nn.Sequential(*model['down1'])

        model['up2'] = [
            nn.Conv2d(self.ngf*4, self.ngf*2, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*2),
            nn.ReLU(True),
            nn.Conv2d(self.ngf*2, self.ngf*2, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf*2, self.ngf, kernel_size=2, stride=2)
            # norm_layer(self.ngf),
            # nn.ReLU(True)
            ]
        self.up2 = nn.Sequential(*model['up2'])


        model['up1'] = [
            nn.Conv2d(self.ngf*2, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.output_nc, kernel_size=3, padding=1)
            # norm_layer(self.output_nc),
            # nn.ReLU(True)
            ]
        self.up1 = nn.Sequential(*model['up1'])
        
        # self.model = model

    def forward(self, input):
        f1 = self.down1(input)
        f2 = self.down2(f1)
        f3 = torch.cat([f2, self.b(f2)], 1)
        f4 = torch.cat([f1, self.up2(f3)], 1)
        return self.up1(f4)

class Unet(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet, self).__init__()
        self.gpu_ids = gpu_ids
        self.model = Unet_struc(opt, input_nc, output_nc, ngf, norm_layer, use_dropout, gpu_ids)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

class Unet_PCA(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet_PCA, self).__init__()
        self.gpu_ids = gpu_ids
        Unet = Unet_struc(opt, opt.PCA_n*2, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        PCAnet = nn.Conv2d(input_nc, opt.PCA_n*2, kernel_size=1, padding=0, bias=False)
        self.model = nn.Sequential(PCAnet,Unet)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

class Unet_struc_small(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet_struc_small, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        model = {}

        model['b'] = [
            nn.MaxPool2d(2), 
            nn.Conv2d(self.ngf*2, self.ngf*4, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf*4, self.ngf*2, kernel_size=2, stride=2)
            # norm_layer(self.ngf*2),
            # nn.ReLU(True)
            ]
        self.b = nn.Sequential(*model['b'])

        model['down2'] = [
            nn.MaxPool2d(2), 
            nn.Conv2d(self.ngf, self.ngf*2, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*2),
            nn.ReLU(True),
            
            ]
        self.down2 = nn.Sequential(*model['down2'])

        model['down1'] = [
            nn.Conv2d(self.input_nc, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            
            ]
        self.down1 = nn.Sequential(*model['down1'])

        model['up2'] = [
            nn.Conv2d(self.ngf*4, self.ngf*2, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf*2, self.ngf, kernel_size=2, stride=2)
            # norm_layer(self.ngf),
            # nn.ReLU(True)
            ]
        self.up2 = nn.Sequential(*model['up2'])


        model['up1'] = [
            nn.Conv2d(self.ngf*2, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.output_nc, kernel_size=3, padding=1)
            # norm_layer(self.output_nc),
            # nn.ReLU(True)
            ]
        self.up1 = nn.Sequential(*model['up1'])
        
        # self.model = model

    def forward(self, input):
        f1 = self.down1(input)
        f2 = self.down2(f1)
        f3 = torch.cat([f2, self.b(f2)], 1)
        f4 = torch.cat([f1, self.up2(f3)], 1)
        return self.up1(f4)

class Unet_small(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet_small, self).__init__()
        self.gpu_ids = gpu_ids
        Unet = Unet_struc_small(opt, input_nc, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        self.model = nn.Sequential(Unet)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

class Unet_PCA_small(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet_PCA_small, self).__init__()
        self.gpu_ids = gpu_ids
        Unet = Unet_struc_small(opt, opt.PCA_n*2, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        PCAnet = nn.Conv2d(input_nc, opt.PCA_n*2, kernel_size=1, padding=0, bias=False)
        self.model = nn.Sequential(PCAnet,Unet)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

class Unet_struc_convconn(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet_struc_convconn, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        model = {}

        model['b'] = [
            nn.MaxPool2d(2), 
            nn.Conv2d(self.ngf*2, self.ngf*4, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf*4, self.ngf*2, kernel_size=2, stride=2)
            # norm_layer(self.ngf*2),
            # nn.ReLU(True)
            ]
        self.b = nn.Sequential(*model['b'])

        model['down2'] = [
            nn.MaxPool2d(2), 
            nn.Conv2d(self.ngf, self.ngf*2, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*2),
            nn.ReLU(True),
            
            ]
        self.down2 = nn.Sequential(*model['down2'])

        model['down1'] = [
            nn.Conv2d(self.input_nc, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            
            ]
        self.down1 = nn.Sequential(*model['down1'])

        model['up2'] = [
            nn.Conv2d(self.ngf*4, self.ngf*2, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf*2, self.ngf, kernel_size=2, stride=2)
            # norm_layer(self.ngf),
            # nn.ReLU(True)
            ]
        self.up2 = nn.Sequential(*model['up2'])


        model['up1'] = [
            nn.Conv2d(self.ngf*2, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.output_nc, kernel_size=3, padding=1)
            # norm_layer(self.output_nc),
            # nn.ReLU(True)
            ]
        self.up1 = nn.Sequential(*model['up1'])

        model['convconn1'] = [
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True)
            ]
        self.convconn1 = nn.Sequential(*model['convconn1'])

        model['convconn2'] = [
            nn.Conv2d(self.ngf*2, self.ngf*2, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*2),
            nn.ReLU(True)
            ]
        self.convconn2 = nn.Sequential(*model['convconn2'])
        
        # self.model = model

    def forward(self, input):
        f1 = self.down1(input)
        f2 = self.down2(f1)
        f3 = torch.cat([self.convconn2(f2), self.b(f2)], 1)
        f4 = torch.cat([self.convconn1(f1), self.up2(f3)], 1)
        return self.up1(f4)

class Unet_PCA_convconn(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet_PCA_convconn, self).__init__()
        self.gpu_ids = gpu_ids
        Unet = Unet_struc_convconn(opt, opt.PCA_n*2, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        PCAnet = nn.Conv2d(input_nc, opt.PCA_n*2, kernel_size=1, padding=0, bias=False)
        self.model = nn.Sequential(PCAnet,Unet)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

class Unet_struc_multiloss(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet_struc_multiloss, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        model = {}

        self.down1 = nn.Sequential(
            nn.Conv2d(self.input_nc, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True)
            )
        self.downsamp1 = nn.MaxPool2d(2)

        self.down2 = nn.Sequential(
            nn.Conv2d(self.ngf, self.ngf*2, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*2),
            nn.ReLU(True)
            )
        self.downsamp2 = nn.MaxPool2d(2)

        self.b = nn.Sequential(
            nn.Conv2d(self.ngf*2, self.ngf*4, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*4),
            nn.ReLU(True)
            )

        self.upsamp1 = nn.ConvTranspose2d(self.ngf*4, self.ngf*2, kernel_size=2, stride=2)
        self.up1 = nn.Sequential(
            nn.Conv2d(self.ngf*4, self.ngf*2, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*2),
            nn.ReLU(True)
            )

        self.upsamp2 = nn.ConvTranspose2d(self.ngf*2, self.ngf, kernel_size=2, stride=2)
        self.up2 = nn.Sequential(
            nn.Conv2d(self.ngf*2, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True)
            )

        self.out3 = nn.Conv2d(self.ngf, self.output_nc, kernel_size=3, padding=1)
        self.out2 = nn.Conv2d(self.ngf*2, self.output_nc, kernel_size=3, padding=1)
        self.out1 = nn.Conv2d(self.ngf*4, self.output_nc, kernel_size=3, padding=1)
        # self.model = model

    def forward(self, input):

        f1 = self.down1(input)
        f2 = self.down2(self.downsamp1(f1))
        f3 = self.b(self.downsamp2(f2))
        f4 = self.up1(torch.cat([f2, self.upsamp1(f3)],1))
        f5 = self.up2(torch.cat([f1, self.upsamp2(f4)],1))
        output3 = self.out3(f5)
        output2 = self.out2(f4)
        output1 = self.out1(f3)
        return {'3': output3, '2': output2, '1': output1}

class Unet_PCA_multiloss(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet_PCA_multiloss, self).__init__()
        self.gpu_ids = gpu_ids
        Unet = Unet_struc_multiloss(opt, opt.PCA_n*2, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        PCAnet = nn.Conv2d(input_nc, opt.PCA_n*2, kernel_size=1, padding=0, bias=False)
        self.model = nn.Sequential(PCAnet,Unet)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

class Unet_struc_deconv(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet_struc_deconv, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        model = {}

        model['b'] = [
            nn.MaxPool2d(2), 
            nn.Conv2d(self.ngf*2, self.ngf*4, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf*4, self.ngf*2, kernel_size=2, stride=2),
            norm_layer(self.ngf*2),
            nn.ReLU(True)
            ]
        self.b = nn.Sequential(*model['b'])

        model['down2'] = [
            nn.MaxPool2d(2), 
            nn.Conv2d(self.ngf, self.ngf*2, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*2),
            nn.ReLU(True),
            
            ]
        self.down2 = nn.Sequential(*model['down2'])

        model['down1'] = [
            nn.Conv2d(self.input_nc, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            
            ]
        self.down1 = nn.Sequential(*model['down1'])

        model['up2'] = [
            nn.Conv2d(self.ngf*4, self.ngf*2, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf*2, self.ngf, kernel_size=2, stride=2),
            norm_layer(self.ngf),
            nn.ReLU(True)
            ]
        self.up2 = nn.Sequential(*model['up2'])


        model['up1'] = [
            nn.Conv2d(self.ngf*2, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.output_nc, kernel_size=3, padding=1)
            # norm_layer(self.output_nc),
            # nn.ReLU(True)
            ]
        self.up1 = nn.Sequential(*model['up1'])
        
        # self.model = model

    def forward(self, input):
        f1 = self.down1(input)
        f2 = self.down2(f1)
        f3 = torch.cat([f2, self.b(f2)], 1)
        f4 = torch.cat([f1, self.up2(f3)], 1)
        return self.up1(f4)

class Unet_PCA_deconv(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet_PCA_deconv, self).__init__()
        self.gpu_ids = gpu_ids
        Unet = Unet_struc_deconv(opt, opt.PCA_n*2, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        PCAnet = nn.Conv2d(input_nc, opt.PCA_n*2, kernel_size=1, padding=0, bias=False)
        self.model = nn.Sequential(PCAnet,Unet)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

# Different number of downsampling
class Unet_2ds_struc(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet_2ds_struc, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        model = {}

        model['b'] = [
            nn.MaxPool2d(2), 
            nn.Conv2d(self.ngf*2, self.ngf*4, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf*4, self.ngf*2, kernel_size=2, stride=2)
            # norm_layer(self.ngf*2),
            # nn.ReLU(True)
            ]
        self.b = nn.Sequential(*model['b'])

        model['down2'] = [
            nn.MaxPool2d(2), 
            nn.Conv2d(self.ngf, self.ngf*2, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*2),
            nn.ReLU(True),
            
            ]
        self.down2 = nn.Sequential(*model['down2'])

        model['down1'] = [
            nn.Conv2d(self.input_nc, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            
            ]
        self.down1 = nn.Sequential(*model['down1'])

        model['up2'] = [
            nn.Conv2d(self.ngf*4, self.ngf*2, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf*2, self.ngf, kernel_size=2, stride=2)
            # norm_layer(self.ngf),
            # nn.ReLU(True)
            ]
        self.up2 = nn.Sequential(*model['up2'])


        model['up1'] = [
            nn.Conv2d(self.ngf*2, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.output_nc, kernel_size=3, padding=1)
            # norm_layer(self.output_nc),
            # nn.ReLU(True)
            ]
        self.up1 = nn.Sequential(*model['up1'])
        
        # self.model = model

    def forward(self, input):
        f1 = self.down1(input)
        f2 = self.down2(f1)
        f3 = torch.cat([f2, self.b(f2)], 1)
        f4 = torch.cat([f1, self.up2(f3)], 1)
        return self.up1(f4)

class Unet_3ds_struc(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet_3ds_struc, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        model = {}

        model['down1'] = [
            nn.Conv2d(self.input_nc, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            
            ]
        self.down1 = nn.Sequential(*model['down1'])

        model['down2'] = [
            nn.MaxPool2d(2), 
            nn.Conv2d(self.ngf, self.ngf*2, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*2),
            nn.ReLU(True),
            
            ]
        self.down2 = nn.Sequential(*model['down2'])

        model['down3'] = [
            nn.MaxPool2d(2), 
            nn.Conv2d(self.ngf*2, self.ngf*4, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*4),
            nn.ReLU(True),
            
            ]
        self.down3 = nn.Sequential(*model['down3'])


        model['b'] = [
            nn.MaxPool2d(2), 
            nn.Conv2d(self.ngf*4, self.ngf*8, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf*8, self.ngf*4, kernel_size=2, stride=2)
            # norm_layer(self.ngf*2),
            # nn.ReLU(True)
            ]
        self.b = nn.Sequential(*model['b'])

        model['up3'] = [
            nn.Conv2d(self.ngf*8, self.ngf*4, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf*4, self.ngf*2, kernel_size=2, stride=2)
            # norm_layer(self.ngf),
            # nn.ReLU(True)
            ]
        self.up3 = nn.Sequential(*model['up3'])
        

        model['up2'] = [
            nn.Conv2d(self.ngf*4, self.ngf*2, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf*2, self.ngf, kernel_size=2, stride=2)
            # norm_layer(self.ngf),
            # nn.ReLU(True)
            ]
        self.up2 = nn.Sequential(*model['up2'])


        model['up1'] = [
            nn.Conv2d(self.ngf*2, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.output_nc, kernel_size=3, padding=1)
            # norm_layer(self.output_nc),
            # nn.ReLU(True)
            ]
        self.up1 = nn.Sequential(*model['up1'])
        
        # self.model = model

    def forward(self, input):
        fd1 = self.down1(input)
        fd2 = self.down2(fd1)
        fd3 = self.down3(fd2)
        fu3 = torch.cat([fd3, self.b(fd3)], 1)
        fu2 = torch.cat([fd2, self.up3(fu3)], 1)
        fu1 = torch.cat([fd1, self.up2(fu2)], 1)
        return self.up1(fu1)

class Unet_3ds_rcab(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet_3ds_rcab, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        n_rcab = 3
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        model = {}

        model['down1'] = [
            nn.Conv2d(self.input_nc, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            self.make_rcab(ngf, n_rcab),
            nn.ReLU(True)
            
            ]
        self.down1 = nn.Sequential(*model['down1'])

        model['down2'] = [
            nn.MaxPool2d(2), 
            nn.Conv2d(self.ngf, self.ngf*2, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*2),
            nn.ReLU(True),
            self.make_rcab(ngf*2, n_rcab)
            
            ]
        self.down2 = nn.Sequential(*model['down2'])

        model['down3'] = [
            nn.MaxPool2d(2), 
            nn.Conv2d(self.ngf*2, self.ngf*4, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*4),
            nn.ReLU(True),
            self.make_rcab(ngf*4, n_rcab)
            ]
        self.down3 = nn.Sequential(*model['down3'])


        model['b'] = [
            nn.MaxPool2d(2), 
            nn.Conv2d(self.ngf*4, self.ngf*8, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*8),
            nn.ReLU(True),
            self.make_rcab(ngf*8, n_rcab),
            nn.ConvTranspose2d(self.ngf*8, self.ngf*4, kernel_size=2, stride=2)
            # norm_layer(self.ngf*2),
            # nn.ReLU(True)
            ]
        self.b = nn.Sequential(*model['b'])

        model['up3'] = [
            nn.Conv2d(self.ngf*8, self.ngf*4, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*4),
            nn.ReLU(True),
            self.make_rcab(ngf*4, n_rcab),
            nn.ConvTranspose2d(self.ngf*4, self.ngf*2, kernel_size=2, stride=2)
            # norm_layer(self.ngf),
            # nn.ReLU(True)
            ]
        self.up3 = nn.Sequential(*model['up3'])
        

        model['up2'] = [
            nn.Conv2d(self.ngf*4, self.ngf*2, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*2),
            nn.ReLU(True),
            self.make_rcab(ngf*2, n_rcab),
            nn.ConvTranspose2d(self.ngf*2, self.ngf, kernel_size=2, stride=2)
            # norm_layer(self.ngf),
            # nn.ReLU(True)
            ]
        self.up2 = nn.Sequential(*model['up2'])


        model['up1'] = [
            nn.Conv2d(self.ngf*2, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            self.make_rcab(ngf, n_rcab),
            nn.Conv2d(self.ngf, self.output_nc, kernel_size=3, padding=1)
            # norm_layer(self.output_nc),
            # nn.ReLU(True)
            ]
        self.up1 = nn.Sequential(*model['up1'])
        
        # self.model = model

    def make_rcab(self, n_feat, n_block):
        rcab_chain = []
        for i in range(n_block):
            rcab_chain.append(RCAB(n_feat=n_feat))
        return nn.Sequential(*rcab_chain)

    def forward(self, input):
        fd1 = self.down1(input)
        fd2 = self.down2(fd1)
        fd3 = self.down3(fd2)
        fu3 = torch.cat([fd3, self.b(fd3)], 1)
        fu2 = torch.cat([fd2, self.up3(fu3)], 1)
        fu1 = torch.cat([fd1, self.up2(fu2)], 1)
        return self.up1(fu1)


class Unet_3ds_deep(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet_3ds_deep, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_convLayer = 4
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        model = {}

        model['down1'] = [
            nn.Conv2d(self.input_nc, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            self.conv_seq(self.ngf, self.n_convLayer)
            
            ]
        self.down1 = nn.Sequential(*model['down1'])

        model['down2'] = [
            nn.MaxPool2d(2), 
            nn.Conv2d(self.ngf, self.ngf*2, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*2),
            nn.ReLU(True),
            self.conv_seq(self.ngf*2, self.n_convLayer)
            
            ]
        self.down2 = nn.Sequential(*model['down2'])

        model['down3'] = [
            nn.MaxPool2d(2), 
            nn.Conv2d(self.ngf*2, self.ngf*4, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*4),
            nn.ReLU(True),
            self.conv_seq(self.ngf*4, self.n_convLayer)
            
            ]
        self.down3 = nn.Sequential(*model['down3'])


        model['b'] = [
            nn.MaxPool2d(2), 
            nn.Conv2d(self.ngf*4, self.ngf*8, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*8),
            nn.ReLU(True),
            self.conv_seq(self.ngf*8, self.n_convLayer),
            nn.ConvTranspose2d(self.ngf*8, self.ngf*4, kernel_size=2, stride=2)
            # norm_layer(self.ngf*2),
            # nn.ReLU(True)
            ]
        self.b = nn.Sequential(*model['b'])

        model['up3'] = [
            nn.Conv2d(self.ngf*8, self.ngf*4, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*4),
            nn.ReLU(True),
            self.conv_seq(self.ngf*4, self.n_convLayer),
            nn.ConvTranspose2d(self.ngf*4, self.ngf*2, kernel_size=2, stride=2)
            # norm_layer(self.ngf),
            # nn.ReLU(True)
            ]
        self.up3 = nn.Sequential(*model['up3'])
        

        model['up2'] = [
            nn.Conv2d(self.ngf*4, self.ngf*2, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*2),
            nn.ReLU(True),
            self.conv_seq(self.ngf*2, self.n_convLayer),
            
            nn.ConvTranspose2d(self.ngf*2, self.ngf, kernel_size=2, stride=2)
            # norm_layer(self.ngf),
            # nn.ReLU(True)
            ]
        self.up2 = nn.Sequential(*model['up2'])


        model['up1'] = [
            nn.Conv2d(self.ngf*2, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            self.conv_seq(self.ngf, self.n_convLayer),
            nn.Conv2d(self.ngf, self.output_nc, kernel_size=3, padding=1)
            # norm_layer(self.output_nc),
            # nn.ReLU(True)
            ]
        self.up1 = nn.Sequential(*model['up1'])
        
        # self.model = model

    def conv_seq(self, ngf, n_layers):
        m = []
        for k in range(n_layers):
            m.append(conv_block(ngf, ngf))
        return nn.Sequential(*m)


    def forward(self, input):
        fd1 = self.down1(input)
        fd2 = self.down2(fd1)
        fd3 = self.down3(fd2)
        fu3 = torch.cat([fd3, self.b(fd3)], 1)
        fu2 = torch.cat([fd2, self.up3(fu3)], 1)
        fu1 = torch.cat([fd1, self.up2(fu2)], 1)
        return self.up1(fu1)

class Unet_3ds_compact(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet_3ds_compact, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        model = {}

        model['down1'] = [
            nn.Conv2d(self.input_nc, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            
            ]
        self.down1 = nn.Sequential(*model['down1'])

        model['down2'] = [
            nn.MaxPool2d(2), 
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            
            ]
        self.down2 = nn.Sequential(*model['down2'])

        model['down3'] = [
            nn.MaxPool2d(2), 
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            
            ]
        self.down3 = nn.Sequential(*model['down3'])


        model['b'] = [
            nn.MaxPool2d(2), 
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf, self.ngf, kernel_size=2, stride=2)
            # norm_layer(self.ngf*2),
            # nn.ReLU(True)
            ]
        self.b = nn.Sequential(*model['b'])

        model['up3'] = [
            nn.Conv2d(self.ngf*2, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf, self.ngf, kernel_size=2, stride=2)
            # norm_layer(self.ngf),
            # nn.ReLU(True)
            ]
        self.up3 = nn.Sequential(*model['up3'])
        

        model['up2'] = [
            nn.Conv2d(self.ngf*2, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf, self.ngf, kernel_size=2, stride=2)
            # norm_layer(self.ngf),
            # nn.ReLU(True)
            ]
        self.up2 = nn.Sequential(*model['up2'])


        model['up1'] = [
            nn.Conv2d(self.ngf*2, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.output_nc, kernel_size=3, padding=1)
            # norm_layer(self.output_nc),
            # nn.ReLU(True)
            ]
        self.up1 = nn.Sequential(*model['up1'])
        
        # self.model = model

    def forward(self, input):
        fd1 = self.down1(input)
        fd2 = self.down2(fd1)
        fd3 = self.down3(fd2)
        fu3 = torch.cat([fd3, self.b(fd3)], 1)
        fu2 = torch.cat([fd2, self.up3(fu3)], 1)
        fu1 = torch.cat([fd1, self.up2(fu2)], 1)
        return self.up1(fu1)


class Unet_3ds_subpixel(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet_3ds_subpixel, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        downsampling = PixelShuffle_downscale
        upsampling = nn.PixelShuffle

        model = {}

        model['down1'] = [
            nn.Conv2d(self.input_nc, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True)
            
            ]
        self.down1 = nn.Sequential(*model['down1'])

        model['down2'] = [
            downsampling(2), 
            nn.Conv2d(self.ngf*4, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True)
            
            ]
        self.down2 = nn.Sequential(*model['down2'])

        model['down3'] = [
            downsampling(2), 
            nn.Conv2d(self.ngf*4, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True)
            
            ]
        self.down3 = nn.Sequential(*model['down3'])


        model['b'] = [
            downsampling(2), 
            nn.Conv2d(self.ngf*4, self.ngf*4, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*4),
            nn.ReLU(True),
            upsampling(2)
            ]
        self.b = nn.Sequential(*model['b'])

        model['up3'] = [
            nn.Conv2d(int(self.ngf*2), self.ngf*4, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*4),
            nn.ReLU(True),
            upsampling(2)
            ]
        self.up3 = nn.Sequential(*model['up3'])
        

        model['up2'] = [
            nn.Conv2d(self.ngf*2, self.ngf*4, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*4),
            nn.ReLU(True),
            upsampling(2)
            ]
        self.up2 = nn.Sequential(*model['up2'])


        model['up1'] = [
            nn.Conv2d(self.ngf*2, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.output_nc, kernel_size=3, padding=1)
            ]
        self.up1 = nn.Sequential(*model['up1'])
        

    def forward(self, input):
        fd1 = self.down1(input)
        fd2 = self.down2(fd1)
        fd3 = self.down3(fd2)
        fu3 = torch.cat([fd3, self.b(fd3)], 1)
        fu2 = torch.cat([fd2, self.up3(fu3)], 1)
        fu1 = torch.cat([fd1, self.up2(fu2)], 1)
        return self.up1(fu1)


class Unet_3ds_dilate(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet_3ds_dilate, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        model = {}

        model['down1'] = [
            nn.Conv2d(self.input_nc, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            
            ]
        self.down1 = nn.Sequential(*model['down1'])

        model['down2'] = [
            nn.MaxPool2d(2), 
            nn.Conv2d(self.ngf, self.ngf*2, kernel_size=3, padding=2, bias=use_bias, dilation=2),
            norm_layer(self.ngf*2),
            nn.ReLU(True),
            
            ]
        self.down2 = nn.Sequential(*model['down2'])

        model['down3'] = [
            nn.MaxPool2d(2), 
            nn.Conv2d(self.ngf*2, self.ngf*4, kernel_size=3, padding=2, bias=use_bias, dilation=2),
            norm_layer(self.ngf*4),
            nn.ReLU(True),
            
            ]
        self.down3 = nn.Sequential(*model['down3'])


        model['b'] = [
            nn.MaxPool2d(2), 
            nn.Conv2d(self.ngf*4, self.ngf*8, kernel_size=3, padding=2, bias=use_bias, dilation=2),
            norm_layer(self.ngf*8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf*8, self.ngf*4, kernel_size=2, stride=2)
            # norm_layer(self.ngf*2),
            # nn.ReLU(True)
            ]
        self.b = nn.Sequential(*model['b'])

        model['up3'] = [
            nn.Conv2d(self.ngf*8, self.ngf*4, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf*4, self.ngf*2, kernel_size=2, stride=2)
            # norm_layer(self.ngf),
            # nn.ReLU(True)
            ]
        self.up3 = nn.Sequential(*model['up3'])
        

        model['up2'] = [
            nn.Conv2d(self.ngf*4, self.ngf*2, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf*2, self.ngf, kernel_size=2, stride=2)
            # norm_layer(self.ngf),
            # nn.ReLU(True)
            ]
        self.up2 = nn.Sequential(*model['up2'])


        model['up1'] = [
            nn.Conv2d(self.ngf*2, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.output_nc, kernel_size=3, padding=1)
            # norm_layer(self.output_nc),
            # nn.ReLU(True)
            ]
        self.up1 = nn.Sequential(*model['up1'])
        
        # self.model = model

    def forward(self, input):
        fd1 = self.down1(input)
        fd2 = self.down2(fd1)
        fd3 = self.down3(fd2)
        fu3 = torch.cat([fd3, self.b(fd3)], 1)
        fu2 = torch.cat([fd2, self.up3(fu3)], 1)
        fu1 = torch.cat([fd1, self.up2(fu2)], 1)
        return self.up1(fu1)

class Unet_2ds_skip(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet_2ds_skip, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.ds_feat_num = 8
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        

        self.feature_ds = nn.Sequential(
            nn.Conv2d(self.input_nc, self.ds_feat_num, kernel_size=1, padding=0, bias=use_bias),
            norm_layer(self.ds_feat_num),
            nn.ReLU(True)
            )

        model = {}

        model['b'] = [
            nn.MaxPool2d(2), 
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf, self.ngf, kernel_size=2, stride=2)
            # norm_layer(self.ngf*2),
            # nn.ReLU(True)
            ]
        self.b = nn.Sequential(*model['b'])

        model['down2'] = [
            nn.MaxPool2d(2), 
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            
            ]
        self.down2 = nn.Sequential(*model['down2'])

        model['down1'] = [
            nn.Conv2d(self.ds_feat_num, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            
            ]
        self.down1 = nn.Sequential(*model['down1'])

        model['up2'] = [
            nn.Conv2d(self.ngf*2, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf, self.ngf, kernel_size=2, stride=2)
            # norm_layer(self.ngf),
            # nn.ReLU(True)
            ]
        self.up2 = nn.Sequential(*model['up2'])


        model['up1'] = [
            nn.Conv2d(self.ngf*2, self.input_nc, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.input_nc),
            nn.ReLU(True)
            ]
        self.up1 = nn.Sequential(*model['up1'])

        self.estim = nn.Sequential(
            nn.Conv2d(self.input_nc + self.input_nc, self.input_nc, kernel_size=1, padding=0, bias=use_bias),
            norm_layer(self.input_nc),
            nn.ReLU(True),
            nn.Conv2d(self.input_nc, self.output_nc, kernel_size=1, padding=0)
            )
        
        # self.model = model

    def forward(self, input):
        f0 = self.feature_ds(input)
        f1 = self.down1(f0)
        f2 = self.down2(f1)
        f3 = torch.cat([f2, self.b(f2)], 1)
        f4 = torch.cat([f1, self.up2(f3)], 1)
        f5 = torch.cat([input, self.up1(f4)], 1)
        return self.estim(f5)


class Unet_3ds(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet_3ds, self).__init__()
        self.gpu_ids = gpu_ids
        Unet = Unet_3ds_struc(opt, input_nc, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        self.model = nn.Sequential(Unet)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class Unet_1ds_struc(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet_1ds_struc, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        model = {}

        model['down1'] = [
            nn.Conv2d(self.input_nc, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            
            ]
        self.down1 = nn.Sequential(*model['down1'])

        model['b'] = [
            nn.MaxPool2d(2), 
            nn.Conv2d(self.ngf, self.ngf*2, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf*2, self.ngf, kernel_size=2, stride=2)
            # norm_layer(self.ngf*2),
            # nn.ReLU(True)
            ]
        self.b = nn.Sequential(*model['b'])

        model['up1'] = [
            nn.Conv2d(self.ngf*2, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.output_nc, kernel_size=3, padding=1)
            # norm_layer(self.output_nc),
            # nn.ReLU(True)
            ]
        self.up1 = nn.Sequential(*model['up1'])
        
        # self.model = model

    def forward(self, input):
        fd1 = self.down1(input)
        fu1 = torch.cat([fd1, self.b(fd1)], 1)
        return self.up1(fu1)

class Unet_0ds_struc(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet_0ds_struc, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        model = {}

        model['down1'] = [
            nn.Conv2d(self.input_nc, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            
            ]
        self.down1 = nn.Sequential(*model['down1'])

        model['up1'] = [
            nn.Conv2d(self.ngf, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.output_nc, kernel_size=3, padding=1)
            # norm_layer(self.output_nc),
            # nn.ReLU(True)
            ]
        self.up1 = nn.Sequential(*model['up1'])
        
        # self.model = model

    def forward(self, input):
        fd1 = self.down1(input)
        return self.up1(fd1)

class Unet_4ds_struc(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet_4ds_struc, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        model = {}

        model['down1'] = [
            nn.Conv2d(self.input_nc, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            
            ]
        self.down1 = nn.Sequential(*model['down1'])

        model['down2'] = [
            nn.MaxPool2d(2), 
            nn.Conv2d(self.ngf, self.ngf*2, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*2),
            nn.ReLU(True),
            
            ]
        self.down2 = nn.Sequential(*model['down2'])

        model['down3'] = [
            nn.MaxPool2d(2), 
            nn.Conv2d(self.ngf*2, self.ngf*4, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*4),
            nn.ReLU(True),
            
            ]
        self.down3 = nn.Sequential(*model['down3'])

        model['down4'] = [
            nn.MaxPool2d(2), 
            nn.Conv2d(self.ngf*4, self.ngf*8, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*8),
            nn.ReLU(True),
            
            ]
        self.down4 = nn.Sequential(*model['down4'])

        model['b'] = [
            nn.MaxPool2d(2), 
            nn.Conv2d(self.ngf*8, self.ngf*16, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*16),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf*16, self.ngf*8, kernel_size=2, stride=2)
            # norm_layer(self.ngf*2),
            # nn.ReLU(True)
            ]
        self.b = nn.Sequential(*model['b'])

        model['up4'] = [
            nn.Conv2d(self.ngf*16, self.ngf*8, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf*8, self.ngf*4, kernel_size=2, stride=2)
            # norm_layer(self.ngf),
            # nn.ReLU(True)
            ]
        self.up4 = nn.Sequential(*model['up4'])

        model['up3'] = [
            nn.Conv2d(self.ngf*8, self.ngf*4, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf*4, self.ngf*2, kernel_size=2, stride=2)
            # norm_layer(self.ngf),
            # nn.ReLU(True)
            ]
        self.up3 = nn.Sequential(*model['up3'])
        

        model['up2'] = [
            nn.Conv2d(self.ngf*4, self.ngf*2, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf*2, self.ngf, kernel_size=2, stride=2)
            # norm_layer(self.ngf),
            # nn.ReLU(True)
            ]
        self.up2 = nn.Sequential(*model['up2'])


        model['up1'] = [
            nn.Conv2d(self.ngf*2, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.output_nc, kernel_size=3, padding=1)
            # norm_layer(self.output_nc),
            # nn.ReLU(True)
            ]
        self.up1 = nn.Sequential(*model['up1'])
        
        # self.model = model

    def forward(self, input):
        fd1 = self.down1(input)
        fd2 = self.down2(fd1)
        fd3 = self.down3(fd2)
        fd4 = self.down4(fd3)
        fu4 = torch.cat([fd4, self.b(fd4)], 1)
        fu3 = torch.cat([fd3, self.up4(fu4)], 1)
        fu2 = torch.cat([fd2, self.up3(fu3)], 1)
        fu1 = torch.cat([fd1, self.up2(fu2)], 1)
        return self.up1(fu1)

class Unet_2ds_dense_struc(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet_2ds_dense_struc, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        model = {}

        model['b'] = [
            nn.MaxPool2d(2), 
            nn.Conv2d(self.ngf*3, self.ngf*4, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf*4, self.ngf*2, kernel_size=2, stride=2)
            # norm_layer(self.ngf*2),
            # nn.ReLU(True)
            ]
        self.b = nn.Sequential(*model['b'])

        model['down2'] = [
            nn.MaxPool2d(2), 
            nn.Conv2d(self.ngf, self.ngf*2, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*2),
            nn.ReLU(True),
            
            ]
        self.down2 = nn.Sequential(*model['down2'])

        model['down1'] = [
            nn.Conv2d(self.input_nc, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            
            ]
        self.down1 = nn.Sequential(*model['down1'])

        model['up2'] = [
            nn.Conv2d(self.ngf*5, self.ngf*2, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf*2, self.ngf, kernel_size=2, stride=2)
            # norm_layer(self.ngf),
            # nn.ReLU(True)
            ]
        self.up2 = nn.Sequential(*model['up2'])


        model['up1'] = [
            nn.Conv2d(self.ngf*4, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.output_nc, kernel_size=3, padding=1)
            # norm_layer(self.output_nc),
            # nn.ReLU(True)
            ]
        self.up1 = nn.Sequential(*model['up1'])

        model['down1_ds2'] = [nn.MaxPool2d(2)]
        self.down1_ds2 = nn.Sequential(*model['down1_ds2'])
        model['down2_us2'] = [nn.ConvTranspose2d(self.ngf*2, self.ngf, kernel_size=2, stride=2)]
        self.down2_us2 = nn.Sequential(*model['down2_us2'])
        model['b_us2'] = [nn.ConvTranspose2d(self.ngf*2, self.ngf, kernel_size=2, stride=2)]
        self.b_us2 = nn.Sequential(*model['b_us2'])
        
        # self.model = model

    def forward(self, input):
        f1 = self.down1(input)
        f2 = self.down2(f1)
        f3 = self.b(torch.cat([self.down1_ds2(f1),f2],1))
        f4 = self.up2(torch.cat([self.down1_ds2(f1),f2,f3],1))
        f5 = self.up1(torch.cat([f1,self.down2_us2(f2),self.b_us2(f3),f4],1))
        return f5

class Unet_1by1_HasCopy_2ds_struc(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet_1by1_HasCopy_2ds_struc, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        model = {}

        model['b'] = [
            
            nn.Conv2d(self.ngf*2, self.ngf*2, kernel_size=1, padding=0, bias=use_bias),
            norm_layer(self.ngf*2),
            nn.ReLU(True)
            
            ]
        self.b = nn.Sequential(*model['b'])

        model['down2'] = [
            
            nn.Conv2d(self.ngf, self.ngf*2, kernel_size=1, padding=0, bias=use_bias),
            norm_layer(self.ngf*2),
            nn.ReLU(True),
            
            ]
        self.down2 = nn.Sequential(*model['down2'])

        model['down1'] = [
            nn.Conv2d(self.input_nc, self.ngf, kernel_size=1, padding=0, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            
            ]
        self.down1 = nn.Sequential(*model['down1'])

        model['up2'] = [
            nn.Conv2d(self.ngf*4, self.ngf*1, kernel_size=1, padding=0, bias=use_bias),
            norm_layer(self.ngf*1),
            nn.ReLU(True)
            ]
        self.up2 = nn.Sequential(*model['up2'])


        model['up1'] = [
            nn.Conv2d(self.ngf*2, self.ngf, kernel_size=1, padding=0, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.output_nc, kernel_size=1, padding=0)
            ]
        self.up1 = nn.Sequential(*model['up1'])
        
        # self.model = model

    def forward(self, input):
        f1 = self.down1(input)
        f2 = self.down2(f1)
        f3 = torch.cat([f2, self.b(f2)], 1)
        f4 = torch.cat([f1, self.up2(f3)], 1)
        return self.up1(f4)

class Unet_1by1_NoCopy_2ds_struc(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet_1by1_NoCopy_2ds_struc, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        model = {}

        model['b'] = [
            
            nn.Conv2d(self.ngf*2, self.ngf*2, kernel_size=1, padding=0, bias=use_bias),
            norm_layer(self.ngf*2),
            nn.ReLU(True)
            
            ]
        self.b = nn.Sequential(*model['b'])

        model['down2'] = [
            
            nn.Conv2d(self.ngf, self.ngf*2, kernel_size=1, padding=0, bias=use_bias),
            norm_layer(self.ngf*2),
            nn.ReLU(True),
            
            ]
        self.down2 = nn.Sequential(*model['down2'])

        model['down1'] = [
            nn.Conv2d(self.input_nc, self.ngf, kernel_size=1, padding=0, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            
            ]
        self.down1 = nn.Sequential(*model['down1'])

        model['up2'] = [
            nn.Conv2d(self.ngf*4, self.ngf*1, kernel_size=1, padding=0, bias=use_bias),
            norm_layer(self.ngf*1),
            nn.ReLU(True)
            ]
        self.up2 = nn.Sequential(*model['up2'])


        model['up1'] = [
            nn.Conv2d(self.ngf*2, self.ngf, kernel_size=1, padding=0, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.output_nc, kernel_size=1, padding=0)
            ]
        self.up1 = nn.Sequential(*model['up1'])
        
        # self.model = model

    def forward(self, input):
        f1 = self.down1(input)
        f2 = self.down2(f1)
        f3 = self.b(f2)
        f4 = self.up2(f3)
        return self.up1(f4)


class Unet_T1T2_struc(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet_T1T2_struc, self).__init__()
        self.gpu_ids = gpu_ids
        if opt.dataset == 'single_dataset' or opt.dataset == 'single_dataset_2':
            Unet_T1 = Net_1by1_struc(opt, input_nc, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
            Unet_T2 = Net_1by1_struc(opt, input_nc, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        elif opt.dataset == 'mrf_dataset':
            Unet_T1 = Unet_3ds_struc(opt, input_nc, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
            Unet_T2 = Unet_3ds_struc(opt, input_nc, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        self.model_T1 = Unet_T1
        self.model_T2 = Unet_T2

    def forward(self, input):
        T1 = self.model_T1(input)
        T2 = self.model_T2(input)
        return torch.cat([T1, T2], 1)

class Unet_T1T2(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet_T1T2, self).__init__()
        self.gpu_ids = gpu_ids
        Unet = Unet_T1T2_struc(opt, input_nc, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        self.model = nn.Sequential(Unet)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

class Unet_T1T2_3ds(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet_T1T2_3ds, self).__init__()
        self.gpu_ids = gpu_ids
        Unet_T1 = Unet_3ds_struc(opt, input_nc, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        Unet_T2 = Unet_3ds_struc(opt, input_nc, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        self.model_T1 = Unet_T1
        self.model_T2 = Unet_T2

    def forward(self, input):
        T1 = self.model_T1(input)
        T2 = self.model_T2(input)
        return torch.cat([T1, T2], 1)

class EncodePath_struc(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(EncodePath_struc, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        model = {}

        model['down1'] = [
            nn.Conv2d(self.input_nc, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            
            ]
        self.down1 = nn.Sequential(*model['down1'])

        model['down2'] = [
            nn.MaxPool2d(2), 
            nn.Conv2d(self.ngf, self.ngf*2, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*2),
            nn.ReLU(True),
            
            ]
        self.down2 = nn.Sequential(*model['down2'])

        model['b_down'] = [
            nn.MaxPool2d(2)
            ]
        self.b_down = nn.Sequential(*model['b_down'])

        # self.model = model

    def forward(self, input):
        f1 = self.down1(input)
        f2 = self.down2(f1)
        f3 = self.b_down(f2)
        return {'f1':f1, 'f2':f2, 'f3':f3}

class DecodePath_struc(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(DecodePath_struc, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        model = {}

        model['b_up'] = [
            nn.Conv2d(self.ngf*2, self.ngf*4, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf*4, self.ngf*2, kernel_size=2, stride=2)
            # norm_layer(self.ngf*2),
            # nn.ReLU(True)
            ]
        self.b_up = nn.Sequential(*model['b_up'])

        model['up2'] = [
            nn.Conv2d(self.ngf*4, self.ngf*2, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf*2, self.ngf, kernel_size=2, stride=2)
            # norm_layer(self.ngf),
            # nn.ReLU(True)
            ]
        self.up2 = nn.Sequential(*model['up2'])


        model['up1'] = [
            nn.Conv2d(self.ngf*2, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.output_nc, kernel_size=3, padding=1)
            # norm_layer(self.output_nc),
            # nn.ReLU(True)
            ]
        self.up1 = nn.Sequential(*model['up1'])
        
        # self.model = model

    def forward(self, input):
        f1 = input['f1']
        f2 = input['f2']
        f3 = input['f3']
        f4 = torch.cat([f2, self.b_up(f3)], 1)
        f5 = torch.cat([f1, self.up2(f4)], 1)
        return self.up1(f5)

class Unet_MultiTask_struc(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet_MultiTask_struc, self).__init__()
        self.gpu_ids = gpu_ids
        self.EncodePath = EncodePath_struc(opt, input_nc, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        self.DecodePath_T1 = DecodePath_struc(opt, input_nc, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        self.DecodePath_T2 = DecodePath_struc(opt, input_nc, output_nc, ngf, norm_layer, use_dropout, gpu_ids)

    def forward(self, input):
        f = self.EncodePath(input)
        T1 = self.DecodePath_T1(f)
        T2 = self.DecodePath_T2(f)
        return torch.cat([T1, T2], 1)

class Unet_MultiTask(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet_MultiTask, self).__init__()
        self.gpu_ids = gpu_ids
        model = Unet_MultiTask_struc(opt, input_nc, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        self.model = nn.Sequential(model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

class EncodePath_Lian_struc(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(EncodePath_Lian_struc, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        model = {}

        model['conv_0'] = [
            nn.Conv2d(self.input_nc, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            
            ]
        self.conv_0 = nn.Sequential(*model['conv_0'])

        model['down_1'] = [
            nn.MaxPool2d(2)
            ]
        self.down_1 = nn.Sequential(*model['down_1'])

        model['conv_1_1'] = [
            nn.Conv2d(self.ngf*2, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            
            ]
        self.conv_1_1 = nn.Sequential(*model['conv_1_1'])

        model['conv_1_0'] = [
            nn.Conv2d(self.input_nc, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            
            ]
        self.conv_1_0 = nn.Sequential(*model['conv_1_0'])


        model['down_2'] = [
            nn.MaxPool2d(2)
            ]
        self.down_2 = nn.Sequential(*model['down_2'])

        model['conv_2_0'] = [
            nn.Conv2d(self.input_nc, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            
            ]
        self.conv_2_0 = nn.Sequential(*model['conv_2_0'])

        model['downim_1'] = [
            nn.MaxPool2d(2)
            ]
        self.downim_1 = nn.Sequential(*model['downim_1'])

        model['downim_2'] = [
            nn.MaxPool2d(4)
            ]
        self.downim_2 = nn.Sequential(*model['downim_2'])

        # self.model = model

    def forward(self, input):
        f_0 = self.conv_0(input)
        f_1_1 = self.down_1(f_0)
        f_1_0 = self.conv_1_0(self.downim_1(input))
        f_1 = self.conv_1_1(torch.cat([f_1_1, f_1_0], 1))
        f_2_1 = self.down_2(f_1)
        f_2_0 = self.conv_2_0(self.downim_2(input))
        f_2 = torch.cat([f_2_1, f_2_0], 1)
        return {'f_0':f_0, 'f_1':f_1, 'f_2':f_2}

class DecodePath_Lian_struc(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(DecodePath_Lian_struc, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        model = {}

        model['b_up'] = [
            nn.Conv2d(self.ngf*2, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf, self.ngf, kernel_size=2, stride=2)
            # norm_layer(self.ngf*2),
            # nn.ReLU(True)
            ]
        self.b_up = nn.Sequential(*model['b_up'])

        model['up2'] = [
            nn.Conv2d(self.ngf*2, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf, self.ngf, kernel_size=2, stride=2)
            # norm_layer(self.ngf),
            # nn.ReLU(True)
            ]
        self.up2 = nn.Sequential(*model['up2'])


        model['up1'] = [
            nn.Conv2d(self.ngf*2, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.output_nc, kernel_size=3, padding=1)
            # norm_layer(self.output_nc),
            # nn.ReLU(True)
            ]
        self.up1 = nn.Sequential(*model['up1'])
        
        # self.model = model

    def forward(self, input):
        f1 = input['f_0']
        f2 = input['f_1']
        f3 = input['f_2']
        f4 = torch.cat([f2, self.b_up(f3)], 1)
        f5 = torch.cat([f1, self.up2(f4)], 1)
        return self.up1(f5)

class Lian_struc(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Lian_struc, self).__init__()
        self.gpu_ids = gpu_ids
        self.EncodePath_T1 = EncodePath_Lian_struc(opt, input_nc, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        self.EncodePath_T2 = EncodePath_Lian_struc(opt, input_nc, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        self.DecodePath_T1 = DecodePath_Lian_struc(opt, input_nc, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        self.DecodePath_T2 = DecodePath_Lian_struc(opt, input_nc, output_nc, ngf, norm_layer, use_dropout, gpu_ids)

    def forward(self, input):
        f_T1 = self.EncodePath_T1(input)
        f_T2 = self.EncodePath_T2(input)
        T1 = self.DecodePath_T1(f_T1)
        T2 = self.DecodePath_T2(f_T2)
        return torch.cat([T1, T2], 1)

class Lian_SingleProperty_struc(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Lian_SingleProperty_struc, self).__init__()
        self.gpu_ids = gpu_ids
        self.EncodePath = EncodePath_Lian_struc(opt, input_nc, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        self.DecodePath = DecodePath_Lian_struc(opt, input_nc, output_nc, ngf, norm_layer, use_dropout, gpu_ids)

    def forward(self, input):
        f = self.EncodePath(input)
        return self.DecodePath(f)

class Lian(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Lian, self).__init__()
        self.gpu_ids = gpu_ids
        model = Lian_struc(opt, input_nc, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        self.model = nn.Sequential(model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

'''
class Net_1by1_struc(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Net_1by1_struc, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            
            nn.Conv2d(self.input_nc, self.ngf, kernel_size=1, padding=0, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, padding=0, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, padding=0, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, padding=0, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.output_nc, kernel_size=1, padding=0)
            
            ]
        self.model = nn.Sequential(*model)


    def forward(self, input):
        return self.model(input)
'''

'''
class Net_1by1_struc(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Net_1by1_struc, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            
            nn.Conv2d(self.input_nc, 200, kernel_size=1, padding=0, bias=use_bias),
            norm_layer(200),
            nn.ReLU(True),
            nn.Conv2d(200, 50, kernel_size=1, padding=0, bias=use_bias),
            norm_layer(50),
            nn.ReLU(True),
            nn.Conv2d(50, 34, kernel_size=1, padding=0, bias=use_bias),
            norm_layer(34),
            nn.ReLU(True),
            nn.Conv2d(34, 17, kernel_size=1, padding=0, bias=use_bias),
            norm_layer(17),
            nn.ReLU(True),
            nn.Conv2d(17, self.output_nc, kernel_size=1, padding=0)
            
            ]
        self.model = nn.Sequential(*model)


    def forward(self, input):
        return self.model(input)
'''

'''
class Net_1by1_struc(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Net_1by1_struc, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.ngf = int(opt.temp)


        model = [
        nn.Conv2d(self.input_nc, self.ngf, kernel_size=1, padding=0, bias=use_bias),
        norm_layer(self.ngf),
        nn.ReLU(True)
        ]
        # Originally 4 norm_layer

        for k in range(opt.FNN_depth-1):
            model.append(nn.Conv2d(self.ngf, self.ngf, kernel_size=1, padding=0, bias=use_bias)) 
            model.append(norm_layer(self.ngf))
            model.append(nn.ReLU(True))
        
        model.append(nn.Conv2d(self.ngf, self.output_nc, kernel_size=1, padding=0))

        self.model = nn.Sequential(*model)


    def forward(self, input):
        return self.model(input)
'''

class Net_1by1_struc(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Net_1by1_struc, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # norm_layer = None
        if norm_layer == None:
            use_bias = True

        if opt.FNN_decrease == 1:
            self.ngf = [500, 250, 100, int(opt.num_D)]
        else:
            self.ngf = [int(opt.num_D) for k in range(opt.FNN_depth)]

        
        model = [
        nn.Conv2d(self.input_nc, self.ngf[0], kernel_size=1, padding=0, bias=use_bias),
        norm_layer(self.ngf[0]) if norm_layer else nn.LeakyReLU(1),
        nn.ReLU(True)
        ]
        # Originally 4 norm_layer

        for k in range(opt.FNN_depth-1):
            model.append(nn.Conv2d(self.ngf[k], self.ngf[k+1], kernel_size=1, padding=0, bias=use_bias)) 
            model.append(norm_layer(self.ngf[k+1]) if norm_layer else nn.LeakyReLU(1))
            model.append(nn.ReLU(True))
        
        model.append(nn.Conv2d(self.ngf[-1], self.output_nc, kernel_size=1, padding=0))

        # model = [i for i in model if i is not None]
        self.model = nn.Sequential(*model)


    def forward(self, input):
        return self.model(input)
    
class FNN(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(FNN, self).__init__()
        self.gpu_ids = gpu_ids

        self.model_T1 = Net_1by1_struc(opt, input_nc, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        self.model_T2 = Net_1by1_struc(opt, input_nc, output_nc, ngf, norm_layer, use_dropout, gpu_ids)

    def forward(self, input):
        T1 = self.model_T1(input)
        T2 = self.model_T2(input)
        return torch.cat([T1, T2], 1)
    

class SQ_module(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(SQ_module, self).__init__()
        self.gpu_ids = gpu_ids

        if opt.Unet_struc == '0ds':
            Post_T1 = Unet_0ds_struc(opt, num_D, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
            Post_T2 = Unet_0ds_struc(opt, num_D, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        elif opt.Unet_struc == '1ds':
            Post_T1 = Unet_1ds_struc(opt, num_D, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
            Post_T2 = Unet_1ds_struc(opt, num_D, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        elif opt.Unet_struc == '2ds':
            Post_T1 = Unet_2ds_struc(opt, num_D, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
            Post_T2 = Unet_2ds_struc(opt, num_D, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        elif opt.Unet_struc == '3ds':
            Post_T1 = Unet_3ds_struc(opt, num_D, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
            Post_T2 = Unet_3ds_struc(opt, num_D, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        elif opt.Unet_struc == '4ds':
            Post_T1 = Unet_4ds_struc(opt, num_D, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
            Post_T2 = Unet_4ds_struc(opt, num_D, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        elif opt.Unet_struc == 'Lian':
            Post_T1 = Lian_SingleProperty_struc(opt, num_D, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
            Post_T2 = Lian_SingleProperty_struc(opt, num_D, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        elif opt.Unet_struc == '3ds_dilate':
            Post_T1 = Unet_3ds_dilate(opt, num_D, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
            Post_T2 = Unet_3ds_dilate(opt, num_D, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        elif opt.Unet_struc == 'RCAN':
            Post_T1 = RCAN(opt, num_D)
            Post_T2 = RCAN(opt, num_D)
        elif opt.Unet_struc == 'Unet_2ds_skip':
            Post_T1 = Unet_2ds_skip(opt, num_D, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
            Post_T2 = Unet_2ds_skip(opt, num_D, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        else:
            raise ValueError('Unet_struc not recognized')
    def forward(self, input):
        return self.model(input)

# Uniform net
class UniNet_init(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UniNet_init, self).__init__()
        self.gpu_ids = gpu_ids

        Pre_T1 = Net_1by1_struc(opt, input_nc, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        Pre_T2 = Net_1by1_struc(opt, input_nc, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        Pre_T1 = Pre_T1.model
        Pre_T2 = Pre_T2.model
        Pre_T1._modules.popitem()
        Pre_T2._modules.popitem()

        num_D = Pre_T1[-3].out_channels

        if opt.Unet_struc == '0ds':
            Post_T1 = Unet_0ds_struc(opt, num_D, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
            Post_T2 = Unet_0ds_struc(opt, num_D, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        elif opt.Unet_struc == '1ds':
            Post_T1 = Unet_1ds_struc(opt, num_D, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
            Post_T2 = Unet_1ds_struc(opt, num_D, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        elif opt.Unet_struc == '2ds':
            Post_T1 = Unet_2ds_struc(opt, num_D, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
            Post_T2 = Unet_2ds_struc(opt, num_D, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        elif opt.Unet_struc == '3ds':
            Post_T1 = Unet_3ds_struc(opt, num_D, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
            Post_T2 = Unet_3ds_struc(opt, num_D, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        elif opt.Unet_struc == '4ds':
            Post_T1 = Unet_4ds_struc(opt, num_D, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
            Post_T2 = Unet_4ds_struc(opt, num_D, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        elif opt.Unet_struc == 'Lian':
            Post_T1 = Lian_SingleProperty_struc(opt, num_D, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
            Post_T2 = Lian_SingleProperty_struc(opt, num_D, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        elif opt.Unet_struc == '3ds_dilate':
            Post_T1 = Unet_3ds_dilate(opt, num_D, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
            Post_T2 = Unet_3ds_dilate(opt, num_D, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        elif opt.Unet_struc == 'RCAN':
            Post_T1 = RCAN(opt, num_D)
            Post_T2 = RCAN(opt, num_D)
        elif opt.Unet_struc == 'Unet_2ds_skip':
            Post_T1 = Unet_2ds_skip(opt, num_D, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
            Post_T2 = Unet_2ds_skip(opt, num_D, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        elif opt.Unet_struc == 'Unet_3ds_subpixel':
            Post_T1 = Unet_3ds_subpixel(opt, num_D, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
            Post_T2 = Unet_3ds_subpixel(opt, num_D, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        elif opt.Unet_struc == 'Unet_3ds_compact':
            Post_T1 = Unet_3ds_compact(opt, num_D, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
            Post_T2 = Unet_3ds_compact(opt, num_D, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        elif opt.Unet_struc == 'Unet_3ds_deep':
            Post_T1 = Unet_3ds_deep(opt, num_D, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
            Post_T2 = Unet_3ds_deep(opt, num_D, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        elif opt.Unet_struc == 'Unet_3ds_rcab':
            Post_T1 = Unet_3ds_rcab(opt, num_D, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
            Post_T2 = Unet_3ds_rcab(opt, num_D, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        else:
            raise ValueError('Unet_struc not recognized')

        self.model_T1 = nn.Sequential(Pre_T1,Post_T1)
        self.model_T2 = nn.Sequential(Pre_T2,Post_T2)

    def forward(self, input):
        T1 = self.model_T1(input)
        T2 = self.model_T2(input)
        return torch.cat([T1, T2], 1)




class UniNet_residue_multiOut(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UniNet_residue_multiOut, self).__init__()
        self.gpu_ids = gpu_ids

        self.featExtr_T1 = Net_1by1_struc(opt, input_nc, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        self.featExtr_T2 = Net_1by1_struc(opt, input_nc, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        self.featExtr_T1 = self.featExtr_T1.model
        self.featExtr_T2 = self.featExtr_T2.model
        self.featExtr_T1._modules.popitem()
        self.featExtr_T2._modules.popitem()

        num_D = self.featExtr_T1[-3].out_channels

        self.Coarse_estimate_T1 = nn.Conv2d(num_D, 1, kernel_size=1, padding=0)
        self.Coarse_estimate_T2 = nn.Conv2d(num_D, 1, kernel_size=1, padding=0)
        
        self.n_residue = 2
        spatial_block = Unet_3ds_struc
        self.Residue_estimate_T1 = []
        self.Residue_estimate_T2 = []
        self.Residue_feat_T1 = []
        self.Residue_feat_T2 = []
        for k in range(self.n_residue):
            self.Residue_estimate_T1.append(nn.Sequential(
                norm_layer(num_D),
                nn.ReLU(True),
                nn.Conv2d(num_D, 1, kernel_size=1, padding=0)))
            self.Residue_estimate_T2.append(nn.Sequential(
                norm_layer(num_D),
                nn.ReLU(True),
                nn.Conv2d(num_D, 1, kernel_size=1, padding=0)))
            self.Residue_feat_T1.append(spatial_block(opt, num_D, num_D, ngf, norm_layer, use_dropout, gpu_ids))
            self.Residue_feat_T2.append(spatial_block(opt, num_D, num_D, ngf, norm_layer, use_dropout, gpu_ids))
        
        self.Residue_estimate_T1_seq = nn.Sequential(*self.Residue_estimate_T1)
        self.Residue_estimate_T2_seq = nn.Sequential(*self.Residue_estimate_T2)
        self.Residue_feat_T1_seq = nn.Sequential(*self.Residue_feat_T1)
        self.Residue_feat_T2_seq = nn.Sequential(*self.Residue_feat_T2)

    def forward(self, input):
        feat_T1 = self.featExtr_T1(input)
        feat_T2 = self.featExtr_T2(input)
        tissue_map = []
        Sum_T1map = self.Coarse_estimate_T1(feat_T1)
        Sum_T2map = self.Coarse_estimate_T2(feat_T2)
        tissue_map.append(torch.cat([Sum_T1map, Sum_T2map], 1))

        for k in range(self.n_residue):
            feat_T1 = self.Residue_feat_T1_seq[k](feat_T1)
            feat_T2 = self.Residue_feat_T2_seq[k](feat_T2)
            Residue_T1map = self.Residue_estimate_T1_seq[k](feat_T1)
            Residue_T2map = self.Residue_estimate_T2_seq[k](feat_T2)
            Sum_T1map = Sum_T1map + 0.1 * Residue_T1map
            Sum_T2map = Sum_T2map + 0.1 * Residue_T2map
            tissue_map.append(torch.cat([Sum_T1map, Sum_T2map], 1))

        return tissue_map

class UniNet_residue(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UniNet_residue, self).__init__()
        self.gpu_ids = gpu_ids
        self.model = UniNet_residue_multiOut(opt, input_nc, output_nc, ngf, norm_layer, use_dropout, gpu_ids)

    def forward(self, input):
        return self.model(input)['final']

class UniNet(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UniNet, self).__init__()
        self.gpu_ids = gpu_ids
        net = UniNet_struc(opt, input_nc, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        # self.model = nn.Sequential(UniNet)
        self.model = net

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

# Uniform net
class UniNet_struc(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UniNet_struc, self).__init__()
        self.gpu_ids = gpu_ids
        pretrained = Unet_T1T2(opt, input_nc, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        if opt.PreNetwork_path != None:
            pretrained.load_state_dict(torch.load(opt.PreNetwork_path))
        pretrained_T1 = pretrained.model[0].model_T1.model
        pretrained_T2 = pretrained.model[0].model_T2.model
        pretrained_T1._modules.popitem()
        pretrained_T2._modules.popitem()

        Pre_T1 = pretrained_T1
        Pre_T2 = pretrained_T2
        if opt.Unet_struc == '0ds':
            Post_T1 = Unet_0ds_struc(opt, Pre_T1[-3].out_channels, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
            Post_T2 = Unet_0ds_struc(opt, Pre_T2[-3].out_channels, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        elif opt.Unet_struc == '1ds':
            Post_T1 = Unet_1ds_struc(opt, Pre_T1[-3].out_channels, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
            Post_T2 = Unet_1ds_struc(opt, Pre_T2[-3].out_channels, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        elif opt.Unet_struc == '2ds':
            Post_T1 = Unet_2ds_struc(opt, Pre_T1[-3].out_channels, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
            Post_T2 = Unet_2ds_struc(opt, Pre_T2[-3].out_channels, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        elif opt.Unet_struc == '3ds':
            Post_T1 = Unet_3ds_struc(opt, Pre_T1[-3].out_channels, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
            Post_T2 = Unet_3ds_struc(opt, Pre_T2[-3].out_channels, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        elif opt.Unet_struc == '4ds':
            Post_T1 = Unet_4ds_struc(opt, Pre_T1[-3].out_channels, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
            Post_T2 = Unet_4ds_struc(opt, Pre_T2[-3].out_channels, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        elif opt.Unet_struc == 'Lian':
            Post_T1 = Lian_SingleProperty_struc(opt, Pre_T1[-3].out_channels, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
            Post_T2 = Lian_SingleProperty_struc(opt, Pre_T2[-3].out_channels, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        else:
            raise ValueError('Unet_struc not recognized')

        # Post_T1 = Unet_3ds_struc(opt, Pre_T1[-3].out_channels, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        # Post_T2 = Unet_3ds_struc(opt, Pre_T2[-3].out_channels, output_nc, ngf, norm_layer, use_dropout, gpu_ids)
        Post_T1.apply(weights_init)
        Post_T2.apply(weights_init)
        self.model_T1 = nn.Sequential(Pre_T1,Post_T1)
        self.model_T2 = nn.Sequential(Pre_T2,Post_T2)

    def forward(self, input):
        T1 = self.model_T1(input)
        T2 = self.model_T2(input)
        return torch.cat([T1, T2], 1)

'''
class hoppe_struc(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(hoppe_struc, self).__init__()
        self.input_nc = 1
        self.output_nc = 2
        self.ngf = 32
        self.gpu_ids = gpu_ids
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        
        model = [
            
            nn.Conv3d(self.input_nc, self.ngf, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0), bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.AvgPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)),
            nn.Conv3d(self.ngf, self.ngf*2, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0), bias=use_bias),
            norm_layer(self.ngf*2),
            nn.ReLU(True),
            nn.AvgPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)),
            nn.Conv3d(self.ngf*2, self.ngf*4, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0), bias=use_bias),
            norm_layer(self.ngf*4),
            nn.ReLU(True),
            nn.AvgPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)),
            nn.Conv3d(self.ngf*4, self.output_nc, kernel_size=(8, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))

            ]


        self.model = nn.Sequential(*model)


    def forward(self, input):
        # a = self.model(input)
        # print(a)
        # print(a.size())
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
'''

class hoppe_struc(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(hoppe_struc, self).__init__()
        self.opt = opt
        self.magnitude = False
        if self.magnitude:
            self.input_nc = 1
        else:
            self.input_nc = 2
        self.output_nc = 2
        self.ngf = 32
        self.gpu_ids = gpu_ids
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        
        
        model = [
            
            nn.Conv3d(self.input_nc, self.ngf, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0), bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.AvgPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)),
            nn.Conv3d(self.ngf, self.ngf*2, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0), bias=use_bias),
            norm_layer(self.ngf*2),
            nn.ReLU(True),
            nn.AvgPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)),
            nn.Conv3d(self.ngf*2, self.ngf*4, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0), bias=use_bias),
            norm_layer(self.ngf*4),
            nn.ReLU(True),
            nn.AvgPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)),
            nn.Conv3d(self.ngf*4, self.output_nc, kernel_size=(8, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))

            ]


        self.model = nn.Sequential(*model)


    def forward(self, input):
        magnitude = self.magnitude

        # number of time points
        ntp = int(self.opt.input_nc/2)

        # calculate magnitude
        if magnitude:
            input_mag = input[:,0:ntp,:,:].clone()
            for k in range(ntp):
                input_mag[:,k,:,:] = (input[:,k,:,:] ** 2 + input[:,ntp+k,:,:] ** 2) ** 0.5
            input = input_mag
        

        # Change to 3d Data
        input = input.unsqueeze(1)
        
        # split real and imaginary
        if not magnitude:
            input = torch.cat((input[:,:,0:ntp,:,:],input[:,:,ntp:ntp+ntp,:,:]), dim=1)
        
        # print(input)
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            a = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            a = self.model(input)
        return a[:,:,0,:,:]

class hoppe_ISMRM2018(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(hoppe_ISMRM2018, self).__init__()
        self.opt = opt
        self.magnitude = False
        if self.magnitude:
            self.input_nc = 1
        else:
            self.input_nc = 2
        self.output_nc = 2
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        
        
        model_convpool = [
            
            nn.Conv1d(self.input_nc, 30, 15, 5),
            nn.ReLU(True),
            nn.Conv1d(30, 60, 10, 3),
            nn.ReLU(True),
            nn.Conv1d(60, 150, 5, 2),
            nn.ReLU(True),
            nn.Conv1d(150, 150, 5, 2),
            nn.ReLU(True),
            nn.Conv1d(150, 300, 3, 1),
            nn.ReLU(True),
            nn.Conv1d(300, 300, 3, 1),
            nn.ReLU(True),
            # nn.AvgPool1d(3, 2)

            ]

        in_feat = 600
        model_fc = [
            nn.Linear(in_feat, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 500),
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Linear(500, 300),
            nn.ReLU(True),
            nn.Linear(300, 300),
            nn.ReLU(True),
            nn.Linear(300, self.output_nc)
        ]


        self.model_convpool = nn.Sequential(*model_convpool)
        self.model_fc = nn.Sequential(*model_fc)


    def forward(self, input):
        magnitude = self.magnitude

        # number of time points
        ntp = int(self.opt.input_nc/2)

        input_view = input.permute(1, 0, 2, 3).contiguous().view(2 * ntp, -1).permute(1, 0).contiguous().view(-1, 2, ntp)

        # calculate magnitude
        if magnitude:
            input_view = (input_view[:,0,:] ** 2 + input_view[:,1,:] ** 2) ** 0.5
            input_view = input_view.unsqueeze(1)

        a = self.model_convpool(input_view)
        a = a.view(a.shape[0], -1)
        a = self.model_fc(a)
        a = a.permute(1, 0).contiguous().view(self.output_nc, input.shape[0], input.shape[2], input.shape[3]).permute(1, 0, 2, 3).contiguous()
        return a

class Cohen_struc(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Cohen_struc, self).__init__()
        self.input_nc = input_nc
        self.output_nc = 2
        self.ngf = 300
        self.gpu_ids = gpu_ids
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        
        model = [
            
            nn.Conv2d(self.input_nc, self.ngf, kernel_size=1, padding=0),
            nn.Tanh(),
            nn.Conv2d(self.ngf, self.ngf, kernel_size=1, padding=0),
            nn.Tanh(),
            nn.Conv2d(self.ngf, self.output_nc, kernel_size=1, padding=0),
            nn.Sigmoid()
            
            ]
        self.model = nn.Sequential(*model)


    def forward(self, input):
        return self.model(input)

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
