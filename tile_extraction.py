import slideflow as sf

def tile(tilepx, tileum, nthread, project_root='./sample_slideflow_project'):
    '''
    Default slide tiling in slideflow. 
    For more detailed usage please refer to https://slideflow.dev/slide_processing/
    '''
    P = sf.Project(project_root)
    dataset = P.dataset(tilepx, tileum, sources=['MyDataset'])
    dataset.extract_tiles(buffer='/path/to/nutter', num_threads=nthread)