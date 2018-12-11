from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import os
import collections
import subprocess

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def scan_files(scan_root, prefix=[''], postfix=['']):
    matched_files = []
    for root, dirs, files in os.walk(scan_root):
        for file in files:
            match = False
            for prefix_i in prefix:
                if file.startswith(prefix_i):
                    for postfix_i in postfix:
                        if file.startswith(prefix_i):
                            match = True
            if match:
                print(os.path.join(root,file))
                matched_files.append(os.path.join(root,file))
    return matched_files

def get_vacant_gpu():
    com="nvidia-smi|sed -n '/%/p'|sed 's/|/\\n/g'|sed -n '/MiB/p'|sed 's/ //g'|sed 's/MiB/\\n/'|sed '/\\//d'"
    gpum=subprocess.check_output(com, shell=True)
    gpum=gpum.decode('utf-8').split('\n')
    gpum=gpum[:-1]
    for i,d in enumerate(gpum):
        gpum[i]=int(gpum[i])
    gpu_id=gpum.index(min(gpum))
    if len(gpum)==4:
        gpu_id=3-gpu_id
    return gpu_id

def preprocess_tissue_property(tp):
    basic_normalize = True
    if basic_normalize:
        tp = tp + (tp==0) * 0.1
        return np.stack((tp[:,:,0]/5000, tp[:,:,1]/500), axis=2)

    T1 = tp[:,:,0]
    T2 = tp[:,:,1]
    T1_dict = list(range(60,2001,10)) + list(range(2020,3001,20)) + list(range(3050,3501,50)) + list(range(4000,5001,500))
    T2_dict = list(range(10,201,5)) + list(range(210,301,10)) + list(range(350,501,50))
    for k in range(T1.shape[0]):
        for j in range(T1.shape[1]):
            T1[k,j] = T1_dict.index(T1[k,j])
            T2[k,j] = T2_dict.index(T2[k,j])
    T1 = T1/len(T1_dict)
    T2 = T2/len(T2_dict)

    return np.stack((T1, T2), axis=2)

def inverse_preprocess_tissue_property(tp):
    basic_normalize = True
    if basic_normalize:
        return np.stack((tp[:,:,0]*5000, tp[:,:,1]*500), axis=2)

    T1 = tp[:,:,0]
    T2 = tp[:,:,1]
    T1_dict = list(range(60,2001,10)) + list(range(2020,3001,20)) + list(range(3050,3501,50)) + list(range(4000,5001,500))
    T2_dict = list(range(10,201,5)) + list(range(210,301,10)) + list(range(350,501,50))
    for k in range(T1.shape[0]):
        for j in range(T1.shape[1]):
            T1_dict_index = int(round(T1[k,j]*len(T1_dict)))
            T1_dict_index = max(min(T1_dict_index, len(T1_dict)-1), 0)
            T2_dict_index = int(round(T2[k,j]*len(T2_dict)))
            T2_dict_index = max(min(T2_dict_index, len(T2_dict)-1), 0)
            T1[k,j] = T1_dict[T1_dict_index]
            T2[k,j] = T2_dict[T2_dict_index]
    return np.stack((T1, T2), axis=2)

def print_log(message, file_name):
    print(message)
    with open(file_name, 'at') as log_file:
        log_file.write(message + '\n')