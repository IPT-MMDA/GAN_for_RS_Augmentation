# Generative Adversarial Network Augmentation for Solving the Training Data Imbalance Problem in Crop Classification
Suplementary material for the corresponding paper at the Remote Sensing Letters journal.


Under revision in the Remote Sensing Letters journal
## Intro
The paper presents a novel data augmentation method employing Generative Adversarial Neural Networks (GANs) with pixel-to-pixel transformation (pix2pix). This approach generates realistic synthetic satellite images with artificial ground truth masks, even for rare crop class distributions. It enables the creation of additional minority class samples, enhancing control over training data balance and outperforming traditional augmentation methods.

![uw_scheme2](https://github.com/IPT-MMDA/GAN_for_RS_Augmentation/assets/142814789/4baf33a8-83cc-4e4b-96d2-627e35cb3929)

## Data preparation

Download Sentinel-2 cloud-free composite of the desired region with red, green, blue and near infra-red 10 m from Google Earth Engine and place it at the root folder as well as a reference ground truth map.

Produce cropped data titles:

```bash
export PYTHONPATH=$PYTHONPATH:$(cd utils && pwd)
python3 dataset_constuction/dataset_cut_titles.py
```

Generate eligible files for training process:

`python3 dataset_constuction/dataset_constuction_real.py`

This will make folders `cropped_256x256` with cropped `*.tif` titles and `data` with `*.npz` archives

## GAN training

Run this command from `training` folder to start GAN training:

`./run_gan.sh`

This will produce a bunch of model checkpoint and example imagws at the `data` folder. The best model to use could be determined not only via metrics but with using those example images to evaluate generation quality by eyes. 

## Segmentation models training

### Real Data
To train model on real data only check `training/run_unet.sh` and make sure that variable `TYPE` is set to `"real"`. Then run this file: 
`./training/run_unet.sh`

### Real+GAN Data
To train model on real+GAN data check `training/run_unet.sh` and make sure that variable `TYPE` is set to `"gan"`. Then generate the GAN data using the best generation model from the previous section. To do this set model location at `dataset_constuction/dataset_generation_gan.py` and run it:

```bash
export PYTHONPATH=$PYTHONPATH:$(cd utils && pwd)
python3 dataset_constuction/dataset_generation_gan.py
```

To start training, run the file: 
`./training/run_unet.sh`

### Real+Statistical Data
To train model on real+Statistical data check `training/run_unet.sh` and make sure that variable `TYPE` is set to `"stat"`. Then generate the Statistical data:

```bash
export PYTHONPATH=$PYTHONPATH:$(cd utils && pwd)
python3 dataset_constuction/dataset_generation_stat.py
```

To start training, run the file: 
`./training/run_unet.sh`

### Real+Replace Data
To train model on real+Replace data check `training/run_unet.sh` and make sure that variable `TYPE` is set to `"replace"`. Then generate the Replace data:

```bash
export PYTHONPATH=$PYTHONPATH:$(cd utils && pwd)
python3 dataset_constuction/dataset_generation_replace.py
```

To start training, run the file: 
`./training/run_unet.sh`

## Evaluation

To evaluate al four models check the paths at /home/anton/Downloads/GAN_for_RS_Augmentation-master/eval/compare_models_performance.py file and run it:

```bash
export PYTHONPATH=$PYTHONPATH:$(cd utils && pwd)
python3 eval/compare_models_performance.py
```

## Citation

```
@article{doi:10.1080/2150704X.2023.2275551,
author = {Leonid Shumilo, Anton Okhrimenko, Nataliia Kussul, Sofiia Drozd and Oleh Shkalikov},
title = {Generative adversarial network augmentation for solving the training data imbalance problem in crop classification},
journal = {Remote Sensing Letters},
volume = {14},
number = {11},
pages = {1131-1140},
year = {2023},
publisher = {Taylor & Francis},
doi = {10.1080/2150704X.2023.2275551},
URL = {https://doi.org/10.1080/2150704X.2023.2275551},
eprint = {https://doi.org/10.1080/2150704X.2023.2275551 }
}
```
