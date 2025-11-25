# S2BF: Saliency-Selective Bidirectional Fusion for Facial Expression Recognition

This repository contains the official PyTorch implementation of **S2BF** (Saliency-Selective Bidirectional Fusion for Facial Expression Recognition).The remaining parts of the codebase will be publicly released upon acceptance of this paper.

We evaluate S2BF on **RAF-DB, AffectNet (7/8 classes), FERPlus, and CAER-S**, achieving state-of-the-art or highly competitive results with favorable efficiency.

## Project Structure


A typical layout is:

```text
S2BF/
├── train.py
├── models/
│   ├── S2BF.py      # main S2BF model (S2BF class, SSCF, IGSRA, etc.)
│   └── ...
├── data_preprocessing/
│   ├── dataset_raf.py
│   ├── dataset_affectnet.py
│   ├── dataset_affectnet_8class.py
│   └── ...
├── data/
│   ├── raf-basic/
│   ├── AffectNet/
│   ├── ferplus/
│   └── CAER-S/
├── utils.py
└── checkpoint/
    └── ...          # saved models

```
You can adjust the structure description here to match your actual repo.


## Pretrained Backbones

S2BF uses two backbones:

- **MobileFaceNet** for facial landmarks (frozen)
- **IR-ResNet50** for image appearance

Please download their pretrained weights and place them under:

```
./models/pretrain/mobilefacenet_model_best.pth.tar
./models/pretrain/ir50.pth
```

## Datasets

We support the following FER benchmarks:

- **RAF-DB**
- **AffectNet (7 classes)**
- **AffectNet (8 classes)**
- **FERPlus**
- **CAER-S**



## Pretrained Models

We provide pretrained S2BF checkpoints on five FER benchmarks.  
All models use IR-50 as the image backbone and MobileFaceNet as the landmark backbone.

| Dataset           | #Classes | Top-1 Acc (%) | Checkpoint                                                   | Notes |
| ----------------- | -------- | ------------- | ------------------------------------------------------------ | ----- |
| RAF-DB            | 7        | 92.50         | [s2bf_rafdb.pth](https://pan.baidu.com/s/1si4jRRVR5J3Itz20FIYJIQ) | xzxz  |
| AffectNet (7 cls) | 7        | 67.74         | [s2bf_affectnet7.pth](https://pan.baidu.com/s/10IZpZ60ZdtCMI_-Se98g4g) | xzxz  |
| AffectNet (8 cls) | 8        | 64.05         | [s2bf_affectnet8.pth](https://pan.baidu.com/s/1wb8rY0k12faLUop7xpznZQ) | xzxz  |
| FERPlus           | 8        | 92.02         | [s2bf_ferplus.pth](https://pan.baidu.com/s/1Qe6rtryFakyDjInAX0MvKQ ) | xzxz  |
| CAER-S            | 7        | 93.26         | [s2bf_caers.pth](https://pan.baidu.com/s/11qdVDdSeroxfHS9MovzrYQ) | xzxz  |

