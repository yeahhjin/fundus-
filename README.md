# Fundus Disease Classification using EfficientNet-B0

## 📌 Project Overview
This project focuses on diagnosing **fundus (retinal) diseases** using a deep learning classification model.  
The goal is to classify fundus images into multiple disease categories with an efficient image classification pipeline.

## 📝 Model Selection
- Initially, we planned to use **YOLOv3** for classification.
- After testing, YOLOv3 showed **poor classification performance** on our dataset.
- Therefore, we selected **EfficientNet-B0** due to its **lightweight architecture** and **better classification performance**.

## 🚀 Training Summary
- ✅ **Train accuracy** showed stable convergence with reasonable performance.
- ❗ **Validation accuracy** was relatively low, mainly caused by **poor data quality** (e.g., mislabeled samples, low-resolution images, class imbalance).

| Model            | Train Accuracy | Validation Accuracy | Notes                             |
|------------------|----------------|----------------------|-----------------------------------|
| YOLOv3 (initial) | Poor           | Poor                 | Not suitable for classification   |
| EfficientNet-B0  | Good           | Low                  | Validation limited by data issues |

## 💻 Repository Contents
This repository includes:
- ✅ **Model training code only**
- ❌ Dataset and full pipeline are **not included** due to data size restrictions

## ⚠️ Known Limitations
- **Overfitting tendencies** due to noisy and imbalanced dataset
- Further **data curation and augmentation** are required for optimal validation performance

## 🖼️ Example Image (Optional)
<img width="2037" height="1039" alt="image" src="https://github.com/user-attachments/assets/469f7394-4352-46f4-a564-f8ee0694f2a3" />


## 🛠️ Environment
- Python, PyTorch
- EfficientNet-B0 (ImageNet pre-trained backbone)

---
