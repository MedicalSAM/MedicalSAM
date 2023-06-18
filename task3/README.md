# SAMed

## Overview

We finetune the `vit_b` version of SAM and output labels of organs. This model applies the low-rank-based (LoRA) finetuning strategy to the SAM image encoder and finetunes it together with the prompt encoder and the mask decoder on labeled medical image segmentation datasets. We observed that the utilization of the warmup finetuning strategy and the AdamW optimizer resulted in successful convergence and reduced loss. To improve the accurate recognition of smaller organs, we also performed data augmentation.


## Prerequisites
- Linux (We tested our codes on Ubuntu 18.04)
- Anaconda
- Python 3.7.11
- Pytorch 1.9.1


run the following commands:
```
conda create -n SAMed python=3.7.11
conda activate SAMed
pip install -r requirements.txt
```

Run [preprocess script](preprocess/) on raw Synapse dataset to process and normalize the data for training and testing. Then run the following command to performe downsampling on the training data.

```
python subsample_datasets.py
```

## Quick start

Here are the instructions: 

1. Change the directory to the rootdir of this repository.
2. Prepare testset by running [preprocess script](preprocess/). 
3. Run this commend to test the performance.
```bash
python test.py --is_savenii --output_dir <Your output directory>
```
Check the test results in `<Your output directory>`.


## Training
We use 1 A5000 GPU for training.
1. Prepare the processed training set whose resolution is `224x224` by running [preprocess script](preprocess/) and [subsample_datasets.py](subsample_datasets.py), and put it in `<Your folder>`. 
2. Run this command to train SAMed.
```bash
python train.py --root_path <Your folder> --output <Your output path> --warmup --AdamW 
```
Check the results in `<Your output path>`.

