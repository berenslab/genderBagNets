# genderBagNets
## About
This repository contains the code associated with the MICCAI 2021 paper [Interpretable gender classification from retinal fundus images using BagNets](https://www.medrxiv.org/content/10.1101/2021.06.21.21259243v1)
## Dependencies
All packages required for running the code in the repository are listed in the file _requirements.txt_
## Data
The above code uses data from the UK Biobank(UKB) repository. Due to the Material Transfer Agreement of UK Biobank, we can not share the data, but researchers can apply for access and subsidized fees are available.
## Code 
**BagNet Model** - BagNet-33 model with the best validation performance is provided in _modelstore/bagnet33/UKB_genderNet_bagnet33_imagenet_098_0.835.hdf5_

**Training and evaluation** - For training and evaluation of Inceptionv3 or variants of BagNets (9, 17, 33) on UKB data refer to _train_UKB.py_. 
For evaluation of the trained models refer to _evaluate.py_. 

**Generating saliency maps** - For generating saliency maps refer to the IPython notebook _Heatmaps.ipynb_. 

**Generating tSNE** - For generating tSNE plot on the test set refer to the IPython notebook _tSNE.ipynb_

**Generating kernel density plots** - For generating the Kernel Density Estimates (KDEs) refer to the IPython notebook _KernelDensities.ipynb_. 
