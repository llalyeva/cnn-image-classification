# CNN Image Classification Project

This repository provides a full training and inference pipeline for an image-classification model built in PyTorch.  
It includes:  

- A custom dataset loader with **image caching**  
- A modular CNN architecture with **BatchNorm + ReLU blocks**  
- Training loop with augmentation, class balancing, LR scheduling  
- Automatic model visualization and learning-curve plotting  
- Inference script for batch prediction on PNG images

---

## 1. Project Structure

```text
.
├── dataset.py               # Custom dataset classes and device-aware dataloader
├── network.py               # StandardCNN architecture
├── training.py              # Training pipeline with evaluation and plotting
├── inference.py             # Inference on arbitrary PNG folders
├── model.pt                 # Example trained model
├── model_architecture.png   # Auto-generated architecture visualization
├── learning_curves.png      # Auto-generated learning-curve plots
├── final_files/             # Final selected outputs
├── output_predictions/      # (generated) inference results
└── pyproject.toml
```

---

## 2. Overview of the Working Principle

### **2.1 Dataset Pipeline**

All dataset logic is defined in **`dataset.py`**.

#### **RawCachingImageFolder**

* Wraps `torchvision.datasets.ImageFolder`.
* Loads **all images into memory once** using a multithreaded `ThreadPoolExecutor`.
* Eliminates repeated disk I/O during training → improves throughput.
* Returns `(PIL.Image, label)`.

#### **AugmentedDataset**

* Wraps a raw dataset and applies **Albumentations** transforms.
* Supports synthetic dataset expansion via `times=k`.
* Converts PIL → NumPy → Augmentation → Tensor.

#### **DeviceDataLoader**

* Wraps a standard PyTorch dataloader.
* Moves batches to the assigned device (`cuda`/`cpu`) **inside the iterator**.
* Keeps the training loop clean and readable.

---

### **2.2 Model Architecture**

Defined in **`network.py`**.

The model (`StandardCNN`) follows a classic but robust CNN pattern:

```
[Conv → BN → ReLU] × 2
MaxPool
Dropout
repeat with increasing channels: 32 → 64 → 128 → 256
→ AdaptiveAvgPool2d(1)
→ Fully connected classifier
```

**Important architectural elements:**

* **ConvBNReLU** blocks: normalize feature maps + accelerate convergence.
* **Dropout** between blocks: reduces overfitting.
* **AdaptiveAvgPool2d(1)**: ensures the CNN can handle fixed 64×64 inputs without needing flattening of large feature maps.
* Final classifier:

  * `Linear(256 → 128 → num_classes)`
  * ReLU + Dropout + Linear

The architecture diagram is automatically generated via `torchview`.

---

### **2.3 Training Pipeline**

Defined in **`training.py`**, the process includes:

#### **Steps:**

1. **Load raw dataset** into memory.
2. **Split** into train/validation/test (80/10/10).
3. Apply:

   * **Heavy augmentations** on training set
     (flips, jittering, affine, noise, resize, normalization)
   * **Light transforms** on validation/test sets.
4. **Compute class-balanced weights** (manually provided counts).
5. Initialize:

   * `StandardCNN()`
   * `Adam` optimizer (LR=3e-4, weight decay)
   * `ReduceLROnPlateau` scheduler (monitors validation loss)
   * Weighted cross-entropy loss
6. Training loop:

   * Forward → Loss → Backward → Update
   * Track loss & accuracy per epoch.
7. Validation loop each epoch.
8. After training:

   * Evaluate on test set.
   * Save:

     * `model.pt`
     * `learning_curves.png`
     * `model_architecture.png`

#### **Learning Curves**

`plot_learning_curves()` generates a plot of training & validation losses, allowing you to inspect:

* underfitting/overfitting patterns
* effect of LR scheduling
* convergence stability

---

### **2.4 Inference Pipeline**

Defined in **`inference.py`**.

#### **Working principle:**

1. Load trained model with `state_dict`.
2. Apply the same preprocessing used in validation:

   * Resize to 64×64
   * Normalize (ImageNet means/std)
   * Convert to tensor
3. Recursively scan the dataset directory for `*.png`.
4. Run the forward pass (no gradients).
5. Select class with highest logit.
6. Save all predictions to:

```
output_predictions/predictions.csv
filename, class_id
```

Inference automatically chooses GPU if available.

---

## 3. How to Run

### **Install Dependencies**

```bash
python -m venv .venv
source .venv/bin/activate

pip install -e .
```

---

### **Training**

```bash
python training.py /path/to/dataset_root
```

This will create:

* `model.pt`
* `learning_curves.png`
* `model_architecture.png`

---

### **Inference**

```bash
python inference.py /path/to/images /path/to/model.pt
```

Results saved to:

```
output_predictions/predictions.csv
```

---

## 4. Dataset Requirements

The training and inference rely on **ImageFolder format**:

```text
dataset_root/
├── class_0/
│   ├── image1.png
│   ├── ...
├── class_1/
│   ├── ...
└── ...
```

* Images must be **PNG** for inference (training can use other formats).
* Labels map to directory names.

---

## 5. Customization

### **Change the Number of Classes**

In `network.py`:

```python
class StandardCNN(nn.Module):
    def __init__(self, num_classes=6):
```

### **Modify Augmentations**

In `training.py`, edit:

```python
train_aug = A.Compose([...])
val_tf = A.Compose([...])
```

### **Change Learning Parameters**

Change epoch count, LR, weight decay, etc., in:

```python
opt = optim.Adam(...)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(...)
fit(epochs=30, ...)
```

---

## 6. Summary

This project demonstrates a clean, extensible pipeline for image classification using PyTorch.
It showcases:

* Efficient data loading with caching
* Clean modular CNN design
* Strong augmentation strategy
* Full training/validation/testing workflow
* Automated model visualization
* Ready-to-use inference pipeline
