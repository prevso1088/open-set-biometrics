# Open-Set Biometrics: Beyond Good Closed-Set Models

This is the official GitHub repository for the paper [Open-Set Biometrics: Beyond Good Closed-Set Models](https://arxiv.org/abs/2407.16133) (ECCV 2024).

## 📌 Overview

This repository contains the implementation for training biometric models using objectives aligned with open-set evaluation protocols

## 🚀 Quick Start

The following code demonstrates how to generate dummy embeddings and labels for a biometric model and compute the open-set loss.

```
import random

import torch

from modeling.losses.open_set import OpenSetLoss

n = 128  # Batch size
c = 256  # Number of channels
p = 16  # Number of body parts

# Generate perfect embedding tensor and label tensor
labels = torch.arange(32).repeat(4)
embeddings = torch.nn.functional.one_hot(labels, num_classes=c)
embeddings = embeddings.unsqueeze(-1).repeat((1, 1, p)).float()

# Generate random embedding tensor and label tensor
embeddings = torch.randn(n, c, p).cuda()
labels = torch.randint(0, 32, (n,)).cuda()

# Parameters
S1 = 32  # Number of subjects to select
S2 = 4  # Number of sequences per subject to select

# Create a list of unique subjects
unique_subjects = list(set(labels))

# Select S1 subjects without replacement
selected_subjects = random.sample(unique_subjects, S1)

# Initialize lists to store selected sequences
selected_sequences = []

# Iterate over selected subjects and select S2 sequences per subject with replacement
for subject in selected_subjects:
    subject_sequences = [i for i, label in enumerate(labels) if label == subject]
    selected_sequences.extend(random.choices(subject_sequences, k=S2))

# Select the corresponding features, labels, cams, and time_seqs based on selected sequences
selected_features = embeddings[selected_sequences]
selected_labels = [unique_subjects.index(labels[i]) for i in selected_sequences]
embeddings, labels = torch.as_tensor(selected_features), torch.as_tensor(selected_labels)

open_set_loss = OpenSetLoss().cuda()(embeddings.cuda(), labels.cuda())
```

## 🛠️ Installation

To set up the environment, please refer to the [OpenGait](https://github.com/ShiqiYu/OpenGait) repository.  
The expected data structure is as follows:
```
/{dataset_root}/{subject_id}/{seq_id}/{cam_id}/sils_64x44.pkl
```

## 🏃‍♀️ Training and Testing

To run training or testing, execute the following commands:

```bash
bash train.sh  # For training
bash test.sh   # For testing
```

## 💻 Citation

If you find this work helpful, please consider citing our paper:

```bibtex
@inproceedings{ open-set-biometrics-beyond-good-closed-set-models,
  author    = {Yiyang Su and Minchul Kim and Feng Liu and Anil Jain and Xiaoming Liu},
  title     = {Open-Set Biometrics: Beyond Good Closed-Set Models},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  address   = {Milan, Italy},
  month     = {October},
  year      = {2024},
}
```

## 🙏 Acknowledgments

We appreciate the authors of [OpenGait](https://github.com/ShiqiYu/OpenGait), upon which this repo is based.
