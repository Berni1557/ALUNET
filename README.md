# Active Learning with the nnUNet and Sample Selection with Uncertainty-Aware Submodular Mutual Information Measure

This repository contains code of the experiments conduced for the paper titled 'Active Learning with nnUNet and Sample Selection with Uncertainty-Aware Submodular Mutual Information Measure' of the MIDL2024 conference.

A new version will be available soon.

## Installation
For dependencies please check environment.yml.
The installation has been tested on Ubuntu 18.04.5 LTS.

### Setting up using conda (some packages might not be optional)
```
conda create -n env_alunet python=3.9
conda activate env_alunet
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pytables -c conda-forge
pip install opencv-python
pip install pynrrd
conda install -c anaconda h5py
pip install imblearn
pip install imutils
pip install keyboard
pip install xgboost
conda install -c conda-forge pingouin
pip install torchcontrib
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ submodlib
cd submodlib
pip install .
pip install scikit-learn
git clone https://github.com/decile-team/submodlib.git
pip install kneed
conda install -c conda-forge umap-learn
conda remove krb5 --force -y
conda install krb5==1.15.1 --no-deps -y
```

Install nnUnet following the installation instructions [nnUNet](https://github.com/MIC-DKFZ/nnUNet).

Ensure that the nnUnet path  [Setting up Paths](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md) are set to:
* nnUNet_raw=code_root/nnunet/data/nnunet/nnUNet_raw
* nnUNet_preprocessed=code_root/nnunet/data/nnunet/nnUNet_preprocessed
* nnUNet_results=code_root/nnunet/data/nnunet/nnUNet_results


## Dataset preparation
Please download imaging data (e.g. Hippocampus, Spleen, Liver) from 
[Medical Segmentation Decathlon](http://medicaldecathlon.com/dataaws/) and organize them as following: 

```
code_root/
└── data/
    └── mds/
        ├── Task03_Liver/
        ├── Task04_Hippocampus/
        └── Task09_Spleen/
              ├── imagesTr
              ├── labelsTr
```


## Run experiments
### Run active learning experiment by selecting a dataset, sampling strategy, fold, budget and a function to execute
```
python UNetXAL3D.py --dataset=<dataset> --strategy=<strategy> --fold=<fold> --budget=<budget> --func=<func>
```

### Example: Active learning with the Hippocampus dataset and random sampling
```
python UNetXAL3D.py --dataset=Hippocampus --strategy=RANDOM --fold=0 --budget=100 --func=alcont
```

### Example: Active learning with the Hippocampus dataset and USIMF sampling
```
python UNetXAL3D.py --dataset=Hippocampus --strategy=USIMF --fold=0 --budget=100 --func=alcont
```

### Example: Active learning with the Hippocampus dataset and training fully labeled dataset
```
python UNetXAL3D.py --dataset=Hippocampus --strategy=FULL --fold=0 --func=alfull
```

## Evaluation
After conduction active learning experiments, please check the output folder for trained nnUNet models and validation results.

## Contributing
Bernhard Föllmer\
Charité - Universitätsmedizin Berlin\
Klinik für Radiologie\
Campus Charité Mitte (CCM)\
Charitéplatz 1\
10117 Berlin\
E-Mail: bernhard.foellmer@charite.de\
Tel: +49 30 450 527365\
http://www.charite.de\
