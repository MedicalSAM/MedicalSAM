# <center> How to use MedicalSAM
## <center> Haochen Zhao, PKU YPC
## Config
First of all, please type this command in shell to clone this repo:
```bash
git clone https://github.com/MedicalSAM/MedicalSAM.git
cd MedicalSAM
```
Before start, you should arrange your files like this in dir MedicalSAM you just cloned.
```
├── checkpoints
│   └──  sam_vit_h_4b8939.pth
├── data
│   └──  RawData
│           ├──  Training
│           └──  Testing
└── logs
``````
## Task1: 
If you want to test SAM performace on BTCV with k points as prompt (where k is a number), with no bounding box, use this command:  
```bash
task1_final.py --points_num k --if_box 0
```
If you want to use bounding box, just set --if_box 1.

## Task2:
Assume you have abundant GPU memory.
If you want to fine-tune the model, with learning rate (l), and epoch (e), use the following command to train the model.
```bash
task2_final.py --lr l --epoch e
```