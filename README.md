# MNSA: Multitasking Network with Spatial Attention and Adaptive Feature Enhancer Module for the Classification and Segmentation of Ultra-Fine-Grained Datasets

This repository contains the PyTorch implementation for the paper **"MNSA: Multitasking Network with Spatial Attention and Adaptive Feature Enhancer Module for the Classification and Segmentation of Ultra-Fine-Grained Datasets"**.


---

## ABSTRACT
This paper introduces a novel neural network architecture called MNSA, designed for classification and segmentation tasks on ultra-fine-grained datasets, particularly plant leaves datasets. The proposed architecture, MNSA, simultaneously performs both classification and segmentation without the need for supplementary information, such as attention guidance, privileged data, etc. For example, other approaches in the literature depend on segmentation masks for classification. The proposed method, MNSA, utilizes a shared backbone to simultaneously and independently compute both segmentation and classification tasks. The classification head of this new architecture integrates a spatial attention mechanism with a fully connected single-layer perceptron for classification. The segmentation head consists of a brand new decoding mechanism with Deformable and Atrous convolution layers with adaptive widths. We integrated a Neural Architecture Search (NAS) algorithm to tune the widths of
these layers to enhance the feature representation and capture fine details in segmentation. We compared our method to state-of-the-art techniques using ultra-fine-grained benchmark datasets. Our model demonstrated better performance in segmentation. Integrating spatial attention in our model improved the accuracy of classification beyond that of the state-of-the-art models. Furthermore, we evaluated our model on five other fine-grained classification datasets, where MNSA consistently outperformed the state-of-the-art models.


## Getting Started

### Prerequisites
To ensure a consistent development and runtime environment, we recommend using the provided `Dockerfile`. Build and run the Docker container as follows:
```bash
docker build -t mnsa .
```

### Data Preparation
The dataset can be downloaded from [Google Drive](https://drive.google.com/drive/u/2/folders/10QKsb3v__qpHuMqM96EA40M_M2DeYXN3).

Place your dataset in the `./data` directory before running the training scripts. 

---

## Training Instructions

### Step 1: Train the Classification Head
To train the classification head, use the following command:
```bash
python -u Train.py --mmanet --cls_ild --dataparallel --data_dir ./data --backbone_class 'densenet161'
```

For additional training options and configurations, please refer to the `train_model_cls.sh` script.

### Step 2: Train the Segmentation Head
Once the classification head is trained, train the segmentation head using the following command:
```bash
python -u Train.py --mmanet --seg_ild --freeze_all --dataparallel --data_dir ./data --backbone_class 'densenet161' --model_path ./best_model.pth --unet --transfer_to 0.250
```

For more segmentation training options and configurations, refer to the `train_model_seg.sh` script.

---

## Code Base
This implementation is based on [MGANet](https://github.com/Markin-Wang/MGANet). 

---


## Citation


---


