import os
from os.path import join, basename, exists, dirname
from glob import glob
from tqdm import tqdm
import torch
from cellpose import io, models

def get_cellpose_seg(tiles_dir, out_dir, cellpose_model_path, channels=['CD20', 'CD3', 'CD4', 'CD8', 'cytokeratin']):
    '''
    Cellpose segmentation on tile images.
    '''
    if not exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob(tiles_dir))

    if basename(files[0]).split('_')[0] in channels:
        files = [f for f in files if 'DAPI' in basename(f)]

    model = models.CellposeModel(pretrained_model=cellpose_model_path, gpu=True, device=torch.device('cuda:3'))
    for finpath in tqdm(files):
        foutname = basename(finpath)
        foutpath = join(out_dir, foutname)
        if exists(foutpath): continue
        img = io.imread(finpath)
        masks, flows, styles = model.eval(img, diameter=10, channels=[0,0], model_loaded=True)
        io.masks_flows_to_seg(img, masks, flows, None, foutpath, [0,0])