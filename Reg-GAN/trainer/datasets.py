import glob
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import cv2
import SimpleITK as sitk
import torch.nn.functional as F
from typing import List, Union


tmachannels = ['CD20', 'CD3', 'CD4', 'CD8', 'DAPI', 'cytokeratin']


def normalize(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min()) * 2 - 1

def load_single(filepath, nchannels=3, trans=None, seed=None):
    if nchannels == 1:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(filepath)
    if trans:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        res = trans(img)
        return res
    else:
        if img.shape[0] != nchannels:
            img = img.transpose((2, 0, 1))
        img_norm = cv2.normalize(img, None, -1, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return torch.from_numpy(img_norm.astype(np.float32))

def load_composite(miflist: List, channels: List, trans=None, seed=None):
    nchannels = len(channels)    
    
    layers = []
    for i in range(nchannels):
        layer = cv2.imread(miflist[i], cv2.IMREAD_GRAYSCALE)
        if layer is not None:
            if trans:
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                layer = trans(layer).squeeze()
            else:
                layer = cv2.normalize(layer, None, -1, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                layer = torch.from_numpy(layer.astype(np.float32))
            layers.append(layer)
        else:
            raise OSError(f"Unable to read image {miflist[i]}")
    # img = np.stack(layers, axis=-1)
    img = torch.stack(layers, dim=0)
    return img.unsqueeze(0)


def get_single_dir(directory):
    if not os.path.isdir(directory):
        raise ValueError(f"The given path is not a directory: {directory}")
    files = [os.path.join(directory, file) for file in sorted(os.listdir(directory)) if file.endswith('.png')]
    return files

def get_matching_tiles(tilesdir, channels):
    nchannels = len(channels)
    hetiles = []
    miftiles = [[] for i in range(nchannels)]
    
    for serial in sorted(os.listdir(tilesdir)):
        slide = serial.split('_')[0]
        hedir = os.path.join(tilesdir, serial, serial + '.ome')
        hetiles += get_single_dir(hedir)
        areadir = os.path.join(tilesdir, serial)
        
        idx = 0
        for d in sorted(os.listdir(areadir)):
            if channels and (d.split('_')[0] not in channels): continue
            patchdir = os.path.join(areadir, d)
            miftiles[idx] += get_single_dir(patchdir)
            idx += 1
    return hetiles, miftiles

def get_datasets_comp(tilesdir, valdir, input_nc, output_nc, size, trans=None, val_trans=None):
    if len(tmachannels) == output_nc:
        channels = tmachannels
    else:
        raise ValueError(f"Unavailable number of channels given.")
    
    hetiles_train, miftiles_train = get_matching_tiles(tilesdir, channels)
    hetiles_val, miftiles_val = get_matching_tiles(valdir, channels)
    
    assert len(miftiles_train) == output_nc == len(miftiles_val)
    for i in range(output_nc):
        assert len(hetiles_train) == len(miftiles_train[i])
        assert len(hetiles_val) == len(miftiles_val[i])
    
    
    hetiles_train = np.array([f for f in hetiles_train], dtype=object)
    miftiles_train = np.array([[f for f in sub] for sub in miftiles_train], dtype=object)
    hetiles_val = np.array([f for f in hetiles_val], dtype=object)
    miftiles_val = np.array([[f for f in sub] for sub in miftiles_val], dtype=object)

    train_data = CompositeDataset(hetiles_train, miftiles_train, input_nc, output_nc, size, channels, trans=trans)
    val_data = CompositeDataset(hetiles_val, miftiles_val, input_nc, output_nc, size, channels, trans=val_trans)
    return train_data, val_data

def get_val_dataset(he_dir, input_nc, size, trans):
    hetiles = sorted(glob.glob(he_dir))
    val_data = ValidationDataset(hetiles, input_nc, size, trans)
    return val_data
    

class ValidationDataset(Dataset):
    def __init__(self, fA, input_nc, size, trans=None):
        self.trans = transforms.Compose(trans) if trans else None  
        self.files_A = fA
        self.input_nc = input_nc
        self.size = size

    def __getitem__(self, index):
        seed = np.random.randint(2147483647) 
        hefilepath = self.files_A[index % len(self.files_A)]
        item_A = load_single(hefilepath, self.input_nc, trans=self.trans, seed=seed)
        item_A = F.interpolate(item_A, size=(self.size, self.size)).squeeze(0)
        return {'A': item_A, 'filepath': hefilepath}
    
    def __len__(self):
        return len(self.files_A)

class CompositeDataset(Dataset):
    def __init__(self, fA, fB, input_nc, output_nc, size, channels,
                trans=None, unaligned=False):
        if trans:
            self.trans = transforms.Compose(trans)
        else:
            self.trans = None
        self.files_A = fA
        self.files_B = fB
        assert len(self.files_A) == len(self.files_B[0])

        self.channels = channels
        self.unaligned = unaligned
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.size = size

    def __getitem__(self, index):
        seed = np.random.randint(2147483647) 
        hefilepath = self.files_A[index % len(self.files_A)]
        item_A = load_single(hefilepath, self.input_nc, trans=self.trans, seed=seed)
        item_B = load_composite(self.files_B[:, index % self.files_B.shape[1]], self.channels, trans=self.trans, seed=seed)
        item_A = F.interpolate(item_A, size=(self.size, self.size)).squeeze(0)
        item_B = F.interpolate(item_B, size=(self.size, self.size)).squeeze(0)
        return {'A': item_A, 'B': item_B, 'index': index, 'filepath': hefilepath}
    
    def __len__(self):
        return len(self.files_A)

################################################################################

def load_norm(filepath, nchannels=3):
    if nchannels == 1:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    elif nchannels == 7 and filepath.split('.')[-1] in ('jpg', 'png'):
        og = cv2.imread(filepath)
        img = np.zeros((nchannels, og.shape[0], og.shape[1]), dtype=np.float32)
        for i in range(nchannels):
            img[i,:,:] = og[:,:,i%3]

    elif nchannels == 7:
        img = sitk.ReadImage(filepath)
        img = sitk.GetArrayFromImage(img)
        if len(img.shape) == 2:
            size = min(img.shape)
            length = max(img.shape)
            assert length % size == 0 and length // size == 8
            resized = np.zeros((nchannels, size, size), dtype=np.float32)
            for i in range(nchannels):
                sl = img[i*size:(i+1)*size, :]
                thresh = np.percentile(sl, 99)
                sl[sl > thresh] = thresh
                resized[i,:,:] = sl
            img = resized
    else:
        img = cv2.imread(filepath)
    
    if img.shape[0] != nchannels:
        resized = np.zeros((nchannels, img.shape[0], img.shape[1]), dtype=np.float32)
        for i in range(nchannels):
            resized[i,:,:] = img[:,:,i]
        img = resized
    img_norm = cv2.normalize(img, None, -1, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return torch.from_numpy(img_norm.astype(np.float32))

def get_datasets(sourcepath, targetpath, input_nc, output_nc, size):
    lstA = sorted(glob.glob(f'{sourcepath}/*'))
    lstB = sorted(glob.glob(f'{targetpath}/*'))
    fnamesA = [x.split('/')[-4][-1] + os.path.basename(x).split('.')[0] for x in lstA]
    fnamesB = [x.split('/')[-4][-1] + os.path.basename(x).split('.')[0] for x in lstB]
    intersection = set(fnamesA) & set(fnamesB)
    files_A = [x for x in lstA if x.split('/')[-4][-1] + os.path.basename(x).split('.')[0] in intersection]
    files_B = [x for x in lstB if x.split('/')[-4][-1] + os.path.basename(x).split('.')[0] in intersection]
    assert len(files_A) == len(files_B)
    
    cut = int(len(files_A) * 0.9)
    train_data = ImageDataset(files_A[:cut], files_B[:cut], input_nc, output_nc, size)
    val_data = ImageDataset(files_A[cut:], files_B[cut:], input_nc, output_nc, size)

    return train_data, val_data


class ImageDataset(Dataset):
    def __init__(self, fA, fB, input_nc, output_nc, size, 
                unaligned=False):
        self.files_A = fA
        self.files_B = fB
        assert len(self.files_A) == len(self.files_B)

        self.unaligned = unaligned
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.size = size

    def __getitem__(self, index):
        item_A = load_norm(self.files_A[index % len(self.files_A)], self.input_nc)
        item_B = load_norm(self.files_B[index % len(self.files_B)], self.output_nc)
        item_A = F.interpolate(item_A.unsqueeze(0), size=(self.size, self.size)).squeeze(0)
        item_B = F.interpolate(item_B.unsqueeze(0), size=(self.size, self.size)).squeeze(0)
        
        return {'A': item_A, 'B': item_B}
    
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
