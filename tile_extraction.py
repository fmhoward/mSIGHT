import slideflow as sf

def tile(tilepx, tileum, nthread):
    '''
    Default slide tiling in slideflow. 
    For more detailed usage please refer to slideflow.dev
    '''
    P = sf.Project('./sample_slideflow_project')
    dataset = P.dataset(tilepx, tileum, sources=['MyDataset'])
    dataset.extract_tiles(buffer='/path/to/nutter', num_threads=nthread)