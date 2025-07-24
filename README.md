# MCANet

This is the official repository of **Multi-Scale Cross-Dimensional Attention Network for Gland Segmentation.**

# Prepare data
Detailed data is available [here](https://figshare.com/articles/dataset/EBHISEG/21540159/1?file=38179080). This dataset, named EBHISEG, is a publicly available breast histopathology benchmark released on Figshare by domain experts. All glandular segmentation annotations were conducted and verified by experienced pathologists. As a curated open-access dataset, the ethical approvals, annotation protocols, and inter-rater reliability assessments were managed and ensured by the original authors prior to publication. Therefore, our study directly builds upon this validated and ethically approved resource without involving additional human subject data collection.

### Dataset descriptions
We conduct extensive experiments to evaluate the performance of our model using six real-world datasets: Normal, Polyp, Low-grade IN, High-grade IN, Serrated adenoma, and Adenocarcinoma. As shown in Table, these datasets contain pathological images of colorectal tissue. It includes a variety of lesion types, from normal tissue to different levels of epithelial dysplasia, polyps, serrated adenomas, and adenocarcinomas. There are a total of $4,456$ $2D$ pathological images. Each image in these datasets has a size of $224 \times 224$ pixels, and we split the data into training, validation, and test sets in a $4:4:2$ ratio.

| Class              | Train | Validation | Test | Total |
|:------------------:|:-----:|:----------:|:----:|:-----:|
| Normal            |  30   |     30     |  16  |  76   |
| Polyp             | 190   |    190     |  94  | 474   |
| Low-grade IN      | 256   |    256     | 127  | 639   |
| High-grade IN     |  74   |     74     |  38  | 186   |
| Serrated adenoma  |  23   |     23     |  12  |  58   |
| Adenocarcinoma    | 318   |    318     | 159  | 795   |

### Implementation Detail
We implement all experiments using Python $3.12.4$ and PyTorch $2.3.1$ on a machine equipped with an NVIDIA GeForce RTX 4070 8GB. The image size is $224 \times 224$.

### Evaluation Metrics

In this paper, we evaluate the model's performance using the mean Dice coefficient (mDice), mean intersection over union (mIoU), accuracy (ACC), recall (Rec), and precision (Pre).

The definitions of the metrics are as follows:

$$
\text{mDice} = \frac{2 \times \text{TP}}{2 \times \text{TP} + \text{FP} + \text{FN}},
$$

$$
\text{mIoU} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}},
$$

$$
\text{ACC} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}},
$$

$$
\text{Rec} = \frac{\text{TP}}{\text{TP} + \text{FN}},
$$

$$
\text{Pre} = \frac{\text{TP}}{\text{TP} + \text{FP}},
$$

where TP refers to true positives, FP refers to false positives, TN refers to true negatives, and FN refers to false negatives.


# Some gland image features
![Some gland image features](./figures/Gland_image.png)
Fig1. Our motivation is as follows: the heterogeneity of the glands increases the difficulty of gland segmentation. (a) Normal: normal gland tissue, (b) Low-grade IN: low-grade intraepithelial neoplasia, (c) Adenocarcinoma.

## Model Architecture
The architecture consists of the following components:
![Model Architecture](./figures/MCANet.png)
Fig2. Illustration of the proposed MCANet for gland segmentation. (a) We design parallel multi-scale attention (PMA) to precisely distinguish between the edge areas of the gland image and other areas. We also design cross-dimensional attention (CDA) to interactively model channels and spatial dimensions in complex gland images. (b) We design multi-scale skip connections to capture the local and global features of glands through cascade connection and residual connection.

### 1. **PMA**
   - We propose parallel multi-scale attention (PMA) mechanism, which preserves pixels' spatial position information through the aggregation of cross-channel and multi-scale information to more accurately distinguish the edge region from other regions of glandular images.

### 2. **CDA**
   - We propose a Cross-Dimensional Attention (CDA) mechanism that captures dependencies across the $(C, H)$, $(C, W)$, and $(H, W)$ dimensions through three separate branches. This enables interactive modeling of channel and spatial dimensions in complex gland images to improve the accuracy of gland image segmentation.
![Model Architecture](./figures/PMA_&_CDA.png)
Fig3. The structure of the PMA is shown on the left, while the structure of the CDA is presented on the right.

### 3. **Multi-scale Skip Connection**
   - We design a multi-scale skip connection module to fuse features from different semantic scales. This module not only preserves detailed information but also enhances the model's contextual awareness, enabling effective extraction and fusion of both local and global information for gland image segmentation.
![Multi-scale Skip Connection](./figures/Multiscale_Skip_Connection.png)
Fig4. The structure of the multi-scale skip connection.

# Experiment
We evaluate the effectiveness of MCANet, including MCANet\_C and MCANet\_R, to address the following research questions:
   - RQ1: Does MCANet outperform the baseline models?
   - RQ2: How do the different components of MCANet (e.g., PMA) impact the segmentation performance?
   - RQ3: Can MCANet provide accurate segmentation results?

## Performance Comparison (RQ1)
![Performance Comparison](./figures/Comparison.png)
   - As shown in the figure, the segmentation effects of our model and the baseline model are compared.

## Case Study (RQ2)
![Performance Comparison](./figures/Performance_Comparison.png)
Fig5. The comparison of segmentation performance on different glad datasets, where the vertical axis represents the datasets and the horizontal axis represents the methods.

## Ablation Study (RQ3)
![Ablation Study](./figures/Ablation_Study.png)

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

# Appendix
## Comparison Table of MCANet and Previous Architectures (Part 1)

| Feature/Method                    | UNet                     | UNet++                  | ResUNet++               | MANet                    | MCANet (Ours)                     |
|:----------------------------------:|:------------------------:|:-----------------------:|:-----------------------:|:------------------------:|:---------------------------------:|
| **Key Innovations**               | Symmetric Encoder-Decoder Network | Multi-scale Skip Connections | Residual Skip Connections | Multi-Scale Attention Mechanism | **Multi-Scale Cross-Dimensional Attention** |
| **Attention Mechanism**           | None                     | None                    | None                    | Multi-Attention Mechanism | **PMA and CDA Synergistic Dual Attention** |
| **Edge Handling**                 | Basic Convolution         | Multi-scale Information Fusion | Enhanced Boundary Detection | Focuses on Simple Gland Boundaries | **Superior Boundary Preservation** |
| **Handling of Intra-class Heterogeneity** | Limited                | Limited                 | Limited                 | Enhances Gland Heterogeneity Modeling | **Excellent via Cross-Dimensional Dependencies** |

---

## Comparison Table of MCANet and Previous Architectures (Part 2)

| Feature/Method                    | DA-TransUNet             | Attention UNet          | TransAttUNet             | MCANet (Ours)                     |
|:----------------------------------:|:------------------------:|:-----------------------:|:-----------------------:|:---------------------------------:|
| **Key Innovations**               | Transformer and Attention Mechanisms | Attention Mechanism for Focused Region Segmentation | Transformer-based Attention Mechanism | **Multi-Scale Cross-Dimensional Attention** |
| **Attention Mechanism**           | Dual Attention (Spatial and Channel) | Spatial Attention Mechanism | Multi-Level Attention Mechanism | **PMA and CDA Synergistic Dual Attention** |
| **Edge Handling**                 | Transformer for Boundary Perception | Focuses on Key Regions, struggles with blurred edges | Attention Mechanism for Boundary Refinement | **Superior Boundary Preservation** |
| **Handling of Intra-class Heterogeneity** | Models Inter-dimensional Dependencies | Improves Focus but Struggles with Intra-class Variations | Models Complex Structures through Attention | **Excellent via Cross-Dimensional Dependencies** |

## Attention Distribution Map
![Ablation Study](./figures/attention_distributions.png)

## Robustness Evaluation under Additive Gaussian Noise (Simulated OOD Perturbation)

| Noise Level (σ) | MCANet_C Consistency | MCANet_R Consistency |
|------------------|----------------------|----------------------|
| 0.01             | 0.9985               | 0.9977               |
| 0.05             | 0.9664               | 0.9509               |
| 0.10             | 0.8877               | 0.8466               |
| 0.20             | 0.7085               | 0.6371               |

## How MCANet Might Integrate with Clinical Workflows and Adapt to Other Histological or Biomedical Image Types

MCANet can be integrated into clinical workflows by serving as an automated tool for precise gland segmentation, assisting pathologists in reducing manual annotation time and improving diagnostic consistency. The model processes input histological images to generate segmentation maps that highlight gland structures, which can be directly used for quantitative analysis or as decision support in pathology labs. Its modular design and efficient inference make it suitable for incorporation into digital pathology pipelines and clinical imaging platforms.

To adapt MCANet for other histological or biomedical image types, the input channels (`n_channels`) and output classes (`n_classes`) can be adjusted to match the specific imaging modality and segmentation targets, such as nuclei, blood vessels, or tumors. Additionally, domain-specific data augmentation and transfer learning strategies can be employed to fine-tune the model on new datasets. The modular attention mechanisms (CDA and PMA) can be customized or scaled according to the complexity and structural characteristics of the target images, enabling broad applicability across various biomedical imaging tasks.


