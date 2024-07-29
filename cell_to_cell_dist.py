
import numpy as np
import pandas as pd
import os
from os.path import join, exists, basename
from glob import glob
from tqdm import tqdm
from scipy.spatial.distance import cdist, pdist
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import gaussian_kde
import plotly.graph_objects as go

def find_adjacent(df_path, outpath, channels, distance=19.69, ):
    '''
    For each cell within a tile, find if the cell is adjacent to cells of a different type.
    '''
    def compute_counts(row, distance):
        counts = {}
        df_tile = df[df['tile'] == row['tile']]
        df_tile = df_tile[df_tile['cell_id'] != row['cell_id']]

        for chan in channels:
            channels = [c for c in channels if c != chan]
            
            if row[chan] == 1:
                count_for_chan = {}
                cell_distance = cdist(row[['centroid_x', 'centroid_y']].to_numpy(dtype=float).reshape(1, -1), df_tile[['centroid_x', 'centroid_y']].to_numpy(dtype=float), metric='euclidean') 
                cell_distance = cell_distance.flatten() <= distance
                for target_col in channels:
                    count_for_chan[target_col] = df_tile[cell_distance & (df_tile[target_col] == 1)].shape[0] > 0
                counts[chan] = count_for_chan
        
        return counts
    
    df = pd.read_parquet(df_path)
    df['centroid_x'] = df['centroid_x'].astype(float)
    df['centroid_y'] = df['centroid_y'].astype(float)

    allcounts = df.apply(lambda row: compute_counts(row, distance), axis=1)
    for col in channels:
        for target_col in channels:
            if col != target_col:
                new_col_name = f"{col}_{target_col}"
                df[new_col_name] = allcounts.apply(lambda x: x.get(col, {}).get(target_col, 0)).astype(int)

    df.to_parquet(outpath, index=False)


def aggregate(df, outpath):
    '''
    Compute slide level metrics from single cell density and adjacency.
    '''
    cols = ['CD20', 'CD4', 'CD8', 'cytokeratin', 'CD20_CD4', 'CD20_CD8', 'CD20_cytokeratin', 'CD4_CD20', 'CD4_CD8', 'CD4_cytokeratin', 
            'CD8_CD20', 'CD8_CD4', 'CD8_cytokeratin', 'cytokeratin_CD20', 'cytokeratin_CD4', 'cytokeratin_CD8']
    df_agg = df.groupby(['slide', 'tile'])[cols].sum().reset_index().groupby('slide').mean(numeric_only=True).reset_index()
    df_agg.to_parquet(outpath, index=False)