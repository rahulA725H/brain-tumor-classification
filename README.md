# Brain Tumor Classification Using Deep Learning

An AI-assisted brain MRI classification project using **PyTorch and ResNet18** to distinguish between MRI images with and without tumors.

> **Disclaimer:** This is an educational and research project, not a clinically validated diagnostic system. It must not be used for medical decisions.

## Project Overview

This project explores how deep learning can be applied to brain MRI image classification.

The workflow includes:

- Dataset inspection and preparation
- Duplicate-image detection and data cleaning
- Grayscale image preprocessing
- ResNet18 model development
- Five-fold stratified group cross-validation
- Model training and checkpoint selection
- Performance evaluation and error analysis

The model performs binary classification:

| Label | Class |
|---|---|
| 0 | No Tumor |
| 1 | Tumor |

## Dataset

The project uses a brain MRI dataset containing two classes.

### Dataset cleaning

The original dataset contained 10,093 images:

| Dataset | Original images |
|---|---:|
| Training | 8,277 |
| Testing | 1,816 |
| **Total** | **10,093** |

Duplicate-image analysis was performed using SHA-256 hashes to identify repeated images and overlap between the original training and testing datasets.

After cleaning:

| Dataset | Cleaned images |
|---|---:|
| Training | 7,947 |
| Testing | 1,575 |
| **Total** | **9,522** |

### Image preprocessing

The preprocessing pipeline includes:

1. Converting images to grayscale.
2. Resizing images to 224 × 224 pixels.
3. Normalizing pixel values to the range [0, 1].
4. Converting images into PyTorch tensors.

The resulting input tensor shape is:

`(1, 224, 224)`

## Model Architecture

The project uses **ResNet18**, implemented with PyTorch.

The architecture was adapted for this binary classification task:

- First convolutional layer modified for single-channel grayscale images.
- Final fully connected layer modified for two output classes.
- Model weights initialized from scratch (`weights=None`).

## Training Configuration

| Parameter | Configuration |
|---|---|
| Deep learning framework | PyTorch |
| Model architecture | ResNet18 |
| Input dimensions | 224 × 224 |
| Input channels | 1 |
| Batch size | 16 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | Cross-Entropy Loss |
| Validation strategy | 5-fold Stratified Group Cross-Validation |
| Early stopping patience | 5 epochs |
| Checkpoint selection | Validation balanced accuracy |

The best checkpoint from each fold is saved separately.

## Evaluation Results

Five fold-specific models were evaluated on the same cleaned test set of 1,575 images.

### Average performance across five models

| Metric | Average |
|---|---:|
| Accuracy | 96.96% |
| Balanced Accuracy | 97.36% |
| Precision | 99.34% |
| Tumor Recall | 94.93% |
| F1-score | 97.25% |
| ROC-AUC | 99.57% |
| Specificity | 99.41% |
| Average Precision | 99.72% |

These values are averages across five model evaluations, not five independent test-set evaluations.

## Error Analysis and Dataset Limitations

Model evaluation also included an investigation of possible dataset-related biases.

The analysis identified:

- Some images were consistently misclassified across all five models.
- Image brightness and original image dimensions differed between correctly classified and missed images.
- Image resolution was strongly associated with class labels in the dataset.

A decision-tree analysis using image width alone achieved a balanced accuracy of 80.37%. This suggests a potential dataset bias, although it does not prove that the ResNet18 model learned the same shortcut.

These findings highlight why high evaluation scores should not automatically be interpreted as evidence of clinical reliability.

### Grouping limitation

Filename prefixes were used as a proxy for grouping because verified patient identifiers were unavailable. Therefore, patient-level independence cannot be guaranteed.

## Project Structure

```text
brain-tumor-classification/
│
├── data/
│   ├── CSVs/
│   └── brain_tumor_mri/
│
├── sessions/
│   ├── plots/
│   ├── best_model_fold_1.pth
│   ├── best_model_fold_2.pth
│   ├── best_model_fold_3.pth
│   ├── best_model_fold_4.pth
│   └── best_model_fold_5.pth
│
├── args.py
├── data_preparation.py
├── dataset.py
├── evaluate.py
├── main.py
├── model.py
├── trainer.py
├── utils.py
└── README.md
```

The `data` directory contains the dataset and CSV files. The `sessions` directory contains trained checkpoints, evaluation outputs and visualizations.

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- OpenCV
- Scikit-learn
- Matplotlib
- Pillow
- tqdm
- Git and Git LFS

## Installation

### Requirements

- Python 3.10 or later
- pip
- Git
- Git LFS

### Clone the repository

```bash
git clone https://github.com/rahulA725H/brain-tumor-classification.git
cd brain-tumor-classification
```

### Download Git LFS files

```bash
git lfs install
git lfs pull
```

### Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install torch torchvision numpy pandas opencv-python scikit-learn matplotlib pillow tqdm
```

*Note: Package versions and hardware-specific PyTorch installation may need adjustment for your system.*

## Future Improvements

- Investigate and reduce potential dataset bias.
- Compare additional CNN architectures.
- Evaluate generalization using verified patient-level splits.
- Explore data augmentation and model explainability.
- Improve reproducibility and experiment tracking.

## Author

**Rahul Singh**

GitHub: [@rahulA725H](https://github.com/rahulA725H)

---

*Developed as a deep learning and medical image classification research project.*