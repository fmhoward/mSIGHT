import pandas as pd
import os
from os.path import join, basename, exists
import numpy as np
import cv2
from skimage.measure import regionprops
from glob import glob
from concurrent.futures import ProcessPoolExecutor
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier


def cell_metrics(img_dir, cellpose_mask_dir, savedir, channels, njobs=16):
    '''
    Calculate volume of pixel intensity for each cell.
    '''
    if not exists(savedir):
        os.makedirs(savedir, exist_ok=True)
    
    files = []
    allfiles = sorted(glob(img_dir))
    for channel in channels:
        files.append([f for f in allfiles if channel in basename(f)])
    length = len(files[0])
    assert all(len(inner) == length for inner in files)
    
    with ProcessPoolExecutor(njobs) as executor:
        executor.map(process_group, zip(*files), [cellpose_mask_dir]*length, [savedir]*length)

def process_group(fs, cellpose_mask_dir, savedir):
    if 'CD20_561_TNBC_' in fs[0]:
        mask_name = fs[0].replace('CD20_561_TNBC_', '').replace('.png', '_seg.npy')
    else:
        mask_name = fs[0].replace('CD20_', '').replace('.png', '_seg.npy')
    mask_name = basename(mask_name)
    mask_fpath = join(cellpose_mask_dir, mask_name)
    img_fpaths = [f for f in fs]

    stats_single(mask_fpath, img_fpaths, savedir, mask_name.replace('_seg.npy', '.parquet'))
    
def stats_single(mask_fpath, img_fpaths, savedir, fout):
    # read image and mask
    mask = np.load(mask_fpath, allow_pickle=True).item()['masks']
    img_fpaths = sorted(img_fpaths)
    imgs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in img_fpaths]
    labels = [basename(p).split('_')[0] for p in img_fpaths]
    regions = np.unique(mask)
    regions = np.delete(regions, 0)
    rows = []
    
    # parameters
    sigma = 13
    dilation_size = 2 # 1 for cell segmentation, 2 for nuclei
    kernel = np.ones((2*dilation_size+1, 2*dilation_size+1))

    # calculation per cell
    for region in regions:
        dilated_region = cv2.dilate((mask == region).astype(np.uint8), kernel, iterations=1)
        blurred_region = cv2.GaussianBlur(dilated_region.astype(float), kernel.shape, sigma)
        
        # find cell centroid
        props = regionprops(dilated_region)
        if len(props) == 0: continue
        assert len(props) == 1
        centroid = props[0].centroid
        
        row = {
            'cell_id': region,
            'centroid_x': f'{centroid[0]:.3f}',
            'centroid_y': f'{centroid[1]:.3f}',
            'area': np.sum(dilated_region),
        }
        
        # find volume for each channel
        for img, label in zip(imgs, labels):
            if label == 'cytokeratin':
                indices = blurred_region != 0
                blurred_region[indices] = 1 - (blurred_region[indices] * 0.5)
                patch = blurred_region * img
            else:
                patch = blurred_region * img
            row[f'volume_{label}'] = np.sum(patch) / np.sum(blurred_region)
            nonempty = patch[patch != 0]
            if len(nonempty) == 0:
                row[f'std_{label}'] = 0
            else:
                row[f'std_{label}'] = np.std(nonempty)

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet(join(savedir, fout))

def cluster(df, method, n_clusters=5, eps=0.5, min_samples=5):
    '''
    Clustering of ground truth cells using volume and standard deviation of pixel intensity.
    Labels needs to be manually asigned to each cluster.
    '''
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=56)
    elif method == 'hierarchical':
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif method == 'dbscan':
        model = DBSCAN(eps=eps, min_samples=min_samples) 
    elif method == 'gmm':
        model = GaussianMixture(n_components=n_clusters)
    res = model.fit_predict(df)
    return res

def classify_fake(df_path, df_ref_path, cluster_col_name, cluster_map, channels=['cytokeratin', 'CD20', 'CD3', 'CD4', 'CD8']):
    '''
    Classify cells on generated images using the clustering results on ground truth cells.
    '''
    df = pd.read_parquet(df_path)
    df_ref = pd.read_parquet(df_ref_path)
    clf = RandomForestClassifier(n_estimators=100, min_samples_split=20, random_state=0)
    cols = [[f'{c}_zscore', f'std_{c}'] for c in channels] 
    cols = [item for sublist in cols for item in sublist]
    clf_name = clf.__class__.__name__
    
    df_ref[cluster_col_name].fillna('NA', inplace=True)
    clf.fit(df_ref[cols], df_ref[cluster_col_name])
    df[clf_name] = clf.predict(df[cols])
    
    for c, label in cluster_map.items():
        df[c] = df[clf_name] == label
    
    channels += ['DAPI']
    for c in [chan for chan in channels if chan not in cluster_map]:
        df[c] = 0
    
    df.to_parquet(df_path.replace('.parquet', f'_{clf_name}.parquet'), index=False)