# mSIGHT
Multiplex Synthetic Immunofluorescence Generated from H&amp;E Transformed Histology


## Software Requirements
Please refer to the yml files for dependencies.
- [Slideflow](https://github.com/jamesdolezal/slideflow)
- [Valis](https://github.com/MathOnco/valis)
- [Cellpose](https://github.com/mouseland/cellpose)


## Steps

### WSI Registration
Registration between pairs of WSIs was done using [Valis](https://github.com/MathOnco/valis), a software developed to register a series of WSIs with rigid and non-rigid transformations. Refer to the [documentation](https://valis.readthedocs.io/en/latest/index.html) for installation and usage.

### Slide Tiling
WSIs were divided into 512 x 512 tiles using [Slideflow](https://github.com/jamesdolezal/slideflow). Refer to the [documentation](https://slideflow.dev) for installation and usage.

### H&E to mIF Translation
Edit the yaml files in Reg-GAN to provide paths to trained model weights and locations for inputs and outputs. Run the model on input files.

### Cell Level Metrics
Perform cell segmentation, classification and cell-to-cell adjacency metrics calculation.

### Outcome Prediction
Evaluate the predictive values of the density and adjacency metrics at patient level.
