# MLP Singer for no7singing dataset

This is a fork of MLP Singer for no7singing dataset (KOKO2022).
Scripts and command are in preparation and needs to be fixed to work well.

## Setup docker
1. Build docker in the docker-mlpsinger 
2. Install packages required to run mlpsinger

## Dataset Preprocess
1. Download no7singing audio and label dataset from the following:

2. Do preprocess with the following comand:
```bash
bash do_preprocess_koko.sh
```

## Train MLP-Singer
Train with the following command:
```bash
bash do_train_koko.sh
```

## Prepare dataset for Hifi-GAN
Create mel-spectrogram from the label data with the following command:
```bash
bash do_infer_trainset.sh
```
