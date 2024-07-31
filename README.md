# mSIGHT
Multiplex Synthetic Immunofluorescence Generated from H&amp;E Transformed Histology


## Software Requirements
Please refer to the yml files for dependencies.
- [Slideflow](https://github.com/jamesdolezal/slideflow)
- [Valis](https://github.com/MathOnco/valis)
- [Cellpose](https://github.com/mouseland/cellpose)


## Steps

### WSI Registration
Registration between pairs of WSIs was done using [Valis](https://github.com/MathOnco/valis), a software developed to register a series of WSIs with rigid and non-rigid transformations. Refer to [Valis documentation](https://valis.readthedocs.io/en/latest/index.html) for installation and usage.

A sample code block for registration can be found in `wsi_registration.py`. 

### Slide Tiling
WSIs were divided into 512 x 512 tiles using [Slideflow](https://github.com/jamesdolezal/slideflow). 

A slideflow project needs to be created prior to tile extraction. A sample project is provided in `sample_slideflow_project` to illustrate the structure and components of a project. Sample code for tile extraction can be found in `tile_extraction.py`.

Refer to [slideflow documentation](https://slideflow.dev) for installation and additional usage.

### H&E to mIF Translation
To train a model, edit the yaml files in Reg-GAN based on the architecture of your choice and provide locations for training inputs and targets. Run `train.py` with the `train()` function.

To generate mIF images from a trained model, edit the yaml files to provide paths to trained model weights and run `train.py` with the `evaluate()` function with paths to input/output directories.

### Cell Level Metrics
To perform cell segmentation, choose a cellpose model that suits your application (see [cellpose documentation](https://cellpose.readthedocs.io/en/latest/)) and run `cell_segmentation.py` with paths to input/output directories and paths to cellpose pretrained model. 

Calculate cell level density metrics for both real and generated tiles as illustrated in  `cell_classification.py`. Perform clustering on real cells and manually assign each cluster a label. Fit a classifier on the clustering results and classify cells on generated images.

Calculate cell-to-cell adjacency metrics in `cell_to_cell_dist.py`.

# Data Availability

