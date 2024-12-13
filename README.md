# MCANet

This is the official repository of **Multi-Scale Cross-Dimensional Attention Network for Gland Segmentation.**

# Prepare data
Detailed data is available [here](https://figshare.com/articles/dataset/EBHISEG/21540159/1?file=38179080).

## Model Architecture
The architecture consists of the following components:
[View the PDF](./pdf/MCANet.pdf).

### 1. **PMA**
   - We propose parallel multi-scale attention (PMA) mechanism, which preserves pixels' spatial position information through the aggregation of cross-channel and multi-scale information to more accurately distinguish the edge region from other regions of glandular images.
[View the PDF](./pdf/PMA_&_CDA.pdf).

### 2. **CDA**
   - We propose a Cross-Dimensional Attention (CDA) mechanism that captures dependencies across the $(C, H)$, $(C, W)$, and $(H, W)$ dimensions through three separate branches. This enables interactive modeling of channel and spatial dimensions in complex gland images to improve the accuracy of gland image segmentation.
[View the PDF](./pdf/PMA_&_CDA.pdf).

### 3. **Attention Embedding Fusion**
   - We employ a feature embedding fusion method that effectively combines the original features and attention feature embeddings through weighted summation. This approach leverages the relationships between the edges, morphology, and neighboring tissues of the glands, as well as the spatial relationships within the glands. It enhances feature representation optimization in gland image segmentation, ensuring semantic consistency in complex medical images and accurately segmenting key structures.

### 4. **Multi-scale Skip Connection**
   - We design a multi-scale skip connection module to fuse features from different semantic scales. This module not only preserves detailed information but also enhances the model's contextual awareness, enabling effective extraction and fusion of both local and global information for gland image segmentation.
[View the PDF](./pdf/Multiscale_Skip_Connection.pdf).

## Installation

You can install the necessary dependencies using `pip`:

```bash
pip install torch torchvision
```

## Usage
Here is an example of how to use the MCANet for segmentation:
```bash
import torch
from MCANet_C import MCANet_C
model = MCANet_C(n_channels=3, n_classes=1)
```

```bash
import torch
from MCANet_R import MCANet_R
model = MCANet_R(n_channels=3, n_classes=1)
```
