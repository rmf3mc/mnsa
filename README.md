# MNSA: Multitasking Network with Spatial Attention and Adaptive Feature Enhancer Module for the Classification and Segmentation of Ultra-Fine-Grained Datasets

This repository contains the PyTorch implementation for the paper **"MNSA: Multitasking Network with Spatial Attention and Adaptive Feature Enhancer Module for the Classification and Segmentation of Ultra-Fine-Grained Datasets"**.

MNSA introduces a multitasking neural network that leverages spatial attention and an adaptive feature enhancer module to achieve high performance on ultra-fine-grained classification and segmentation tasks. The model is optimized for datasets with intricate patterns and subtle differences.

---

## Getting Started

### Prerequisites
To ensure a consistent development and runtime environment, we recommend using the provided `Dockerfile`. Build and run the Docker container as follows:
```bash
docker build -t mnsa .
```

### Data Preparation
The dataset can be downloaded from [Google Drive](https://drive.google.com/drive/u/2/folders/10QKsb3v__qpHuMqM96EA40M_M2DeYXN3).

Place your dataset in the `./data` directory before running the training scripts. Ensure the data structure follows the expected format.

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
python -u Train.py --mmanet --seg_ild --freeze_all --dataparallel --data_dir ./data --backbone_class 'densenet161' --model_path best_model.pth --unet --transfer_to 0.250
```

For more segmentation training options and configurations, refer to the `train_model_seg.sh` script.

---

## Code Base
This implementation is based on [MGANet](https://github.com/Markin-Wang/MGANet). 

---


## License
This project is licensed under the [MIT License](LICENSE).

---


