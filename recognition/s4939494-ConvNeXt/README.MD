# Alzheimer's Disease Classification using ConvNeXt

# Overview  

Alzheimer’s disease (AD) is a neurodegenerative disease and is one of the most common form of demnentia. As there are no treatments available to cure Alzheimer's disease, the focus is a lot more about preventing and detecting, which is why early detection is so important [[1]](#ref1).

In this project, our goal is to reach 80% accuracy on classifying MRI brain images as either Alzheimer’s Disease (AD) or Normal Control (NC) using the ConvNeXt-Tiny architecture.

# Model Architecture

The ConvNeXt architecture is a modern convolutional neural network inspired by the design principles of Vision Transformers. It introduces key improvements such as depthwise convolutions, LayerNorm, GELU activations, residual scaling, and a hierarchical stage structure.  

![ConvNeXt Architecture](images/convnext_structure.png)

Figure 1. ConvNeXt general structure [[3]](#ref3)

The ConvNeXt architecture consists of the following components:

### Patch Partitioning:
A 4×4 convolution with stride 4 divides the input image into non-overlapping patches, converting each patch into an embedded feature token. For a 256×256 image, this results in a 64
×64 grid of patch embeddings with 96 channels.

### Hierarchical Feature Representation:
The network consists of 4 stages, each with multiple ConvNeXt blocks. Channel dimensions increase (96 → 192 → 384 → 768), while spatial dimensions decrease through strided convolutions. This hierarchy enables learning of both low-level texture and high-level structural patterns.

![ConvNeXt Block](images/convnext_block.png)

Figure 2. ConvNeXt block [[4]](#ref2)

### Global Average Pooling and Linear Head:
After the final stage, global average pooling condenses features, followed by a linear head producing 2-class logits (AD vs NC).

Due to its balance between efficiency and representational power, ConvNeXt-Tiny is highly suitable for medical imaging tasks requiring structural sensitivity and spatial consistency.

# Dataset Description

The dataset used is the Alzheimer’s Disease Neuroimaging Initiative (ADNI) dataset, which contains MRI scans from patients with Alzheimer’s Disease (AD) and normal controls (NC) classified. Each image is grayscale with an original resolution of 256×240 pixels and is padded to 256×256 for input into the model.

| Split |   AD  |   NC  | Total |
|-------|-------|-------|-------|
| Train | 10,400| 10,820| 21,220|
| Val   | 2,160 | 2,140 | 4,300 |
| Test  | 4,460 | 4,540 | 9,000 |

A patient-wise split ensures that all slices from a single patient remain in the same partition, preventing data leakage and ensuring fair evaluation.

### Example MRI Images
**AD Images**  
![AD Images](images/ADNI/AD.jpeg)|![AD Images](images/ADNI/AD2.jpeg)|![AD Images](images/ADNI/AD3.jpeg)

**NC Images**  
![NC Images](images/ADNI/NC.jpeg)|![NC Images](images/ADNI/NC2.jpeg)|![NC Images](images/ADNI/NC3.jpeg)

# Training Process
___

The model was trained from scratch using PyTorch with AdamW optimisation, CrossEntropyLoss, and CosineAnnealingLR scheduling. Early stopping was used based on validation loss with a patience of 50 epochs.

### Hyperparameters
- Batch Size: 256  
- Learning Rate: 0.004  
- Weight Decay: 0.01  
- Num Epochs: 500  
- Warmup Epochs: 10  
- Patience: 50  
- Label Smoothing: 0.1  
- Drop Path Rate: 0.2

### Transformations

To improve robustness, the following augmentations were used:

1. Padding (256×240 → 256×256)  
2. Random Affine (±7° rotation, ±4% translation, 0.97–1.03 scaling)  
3. Brightness/Contrast jitter (±15%)  
4. Random Gamma (0.9–1.1)  
5. Gaussian Blur (p=0.2)  
6. Small random noise and erasing (p=0.25)  
7. Normalisation (mean=0.5, std=0.5)

Transformations were kept mild to ensure anatomical integrity of MRI images.

# Results

### Performance Metrics

The training and validation losses and accuracies were recorded across epochs, as shown below.

![Training Curves](images/train_val_curves.png)


The best checkpoint achieved ~0.94 validation accuracy.

### Confusion Matrix
![Confusion Matrix](images/confusion_matrix_test.png)

### Classification Report

Accuracy: 0.8013

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| AD | 0.8549 | 0.7215 | 0.7826 | 4460 |
| NC | 0.7628 | 0.8797 | 0.8171 | 4540 |

### Example Predictions




# Usage
To train or evaluate the model, follow these steps.

### 1. Clone the repository

```bash
git clone <https://github.com/MinhQuocLe/PatternAnalysis-2025>
git checkout topic-recognition
cd ./recognition/s4939494-convNeXt
```

### 2. Install dependencies

```bash
pip install sklearn.metrics
-r 
requirements.txt
```


### 3. Adjust data directory
Assuming the dataset has been downloaded and is in its original format,  adjust the file structure to be as follows:

`/home/groups/comp3710/ADNI/AD_NC` or manually change the Hyperparameters data_root in `train.py` 

```
/home/groups/comp3710/ADNI/AD_NC/
    train/
        AD/
            ad_001.jpg
            ad_002.jpg
            ...
        NC/
            nc_001.jpg
            nc_002.jpg
            ...
    test/
        AD/
            ad_001.jpg
            ad_002.jpg
            ...
        NC/
            nc_001.jpg
            nc_002.jpg
            ...

```

### 4. train the model

To train the model, run the following command.

```bash
$ python train.py
```

### 5. Evaluations
To evaluate the accuracy of the model, run the following command.

```bash
$ python predict.py 
```

# References

<a id="ref1">1.</a>Todd S, Barr S, Roberts M, Passmore AP (November 2013). "Survival in dementia and predictors of mortality: a review". International Journal of Geriatric Psychiatry. 28 (11): 1109–1124. doi:10.1002/gps.3946. PMID 23526458.

<a id="ref2">2.</a>Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2021). Swin Transformer: Hierarchical vision transformer using shifted windows. arXiv. https://arxiv.org/abs/2103.14030

<a id ="ref3">3.</a>something https://medium.com/@atakanerdogan305/convnext-next-generation-of-convolutional-networks-325607a08c46