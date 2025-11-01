
# 🧠 CNN Optimizer Performance Comparison: SGD vs Adam

A detailed analysis comparing **SGD** and **Adam** optimizers on **Convolutional Neural Networks (CNNs)** for image classification.  
This project demonstrates how optimizer selection influences training stability, accuracy, and convergence speed using a practical, experiment-based setup.

---

## 📘 Project Overview

This project compares the two most widely used optimizers:
- **SGD (Stochastic Gradient Descent)** with Momentum  
- **Adam (Adaptive Moment Estimation)**  

It follows a clear experimental pipeline — from data preparation to visualization — helping to understand the strengths and weaknesses of each optimizer in CNN training.

---

## 📁 Project Structure
📦 CNN Optimizer Comparison
├── dataset/ # Your image dataset (auto-detected)
├── cnn_optimizer_comparison.ipynb # Main Colab notebook
├── results/ # Generated graphs and results
│ ├── accuracy_comparison.png
│ ├── loss_comparison.png
│ └── optimizer_results.csv
└── README.md # Project documentation

---

## 🚀 Key Features

### 1️⃣ Unified Analysis Framework
- **Dataset Detection:** Works with any custom dataset (or CIFAR-10 by default)  
- **CNN Architecture:** Uses BatchNorm, Dropout, and ReLU layers  
- **Optimizer Comparison:** Trains CNNs with both SGD and Adam  
- **Visualization:** Produces detailed performance plots  

### 2️⃣ Metrics Computed
- Training and validation accuracy/loss  
- Total training time  
- Convergence behavior  
- Generalization and overfitting patterns  

---

## ⚙️ How to Run

### On Google Colab:
1. Upload your dataset ZIP (or skip to use CIFAR-10)  
2. Run all cells in **`cnn_optimizer_comparison.ipynb`**

### Or locally (Python script version):
```bash
python cnn_optimizer_comparison.py

## 📊 Key Findings

| **Optimizer** | **Validation Accuracy** | **Training Time (s)** | **Convergence** |
|----------------|--------------------------|------------------------|-----------------|
| SGD            | Moderate (slower)        | Higher                 | Gradual         |
| Adam           | Higher                   | Faster                 | Stable          |

### Summary
- **Adam** achieves faster convergence and higher accuracy.  
- **SGD** performs well with tuned hyperparameters but converges slower.  
- **Adam** is preferred for general and research use; **SGD** suits production stability.  

---

## 🧠 Methodology

### Dataset Loading & Preprocessing
- Automatically extracts ZIP files and detects class folders.  
- Applies normalization and augmentation (rotation, flips, zooms).  

### Model Architecture
- Convolutional layers → BatchNorm → ReLU  
- MaxPooling + Dropout for regularization  
- Dense + Softmax for classification  

### Training Setup
- Same CNN trained twice (once with SGD, once with Adam)  
- **Epochs:** 5  
- **Batch Size:** 32  
- **Validation Split:** 0.2  

### Evaluation
- Compares both optimizers using identical configurations.  
- Saves accuracy/loss graphs and CSV summary.  

---

## 🖼️ Visualization Outputs

The notebook generates:
- **Accuracy Curve:** SGD vs Adam validation accuracy  
- **Loss Curve:** SGD vs Adam validation loss  
- **Performance Bar Chart:** Accuracy & training time  
- **Result CSV:** Consolidated metrics summary  

All generated files are saved inside the **`results/`** directory.  

---

## 🧰 Technical Details

### Optimizers
- **SGD:** learning_rate = 0.01, momentum = 0.9  
- **Adam:** learning_rate = 0.001  

### Dataset
- Custom uploaded dataset **or** CIFAR-10 fallback.  
- Automatically split into training and validation sets.  

### Model Summary
- **Input shape:** 64×64×3 (adjustable)  
- **Layers:** 3× Conv2D blocks + Pooling + Dropout  
- **Output:** Dense layers for classification  

---

## 🎓 Educational Value

This project is ideal for:
- Understanding **optimizer behavior** in CNNs.  
- Learning **experimental comparison** methods.  
- Practicing **Colab-based deep learning workflows**.  
- Visualizing **optimizer convergence and stability**.  

---

## 🔮 Future Enhancements

- Compare more optimizers (RMSProp, AdamW, AdaGrad).  
- Add learning rate schedulers.  
- Integrate transfer learning models (ResNet, VGG).  
- Evaluate larger datasets like ImageNet or custom datasets.  

---

## 🧾 Dependencies

- TensorFlow / Keras  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  

> 🧩 All dependencies install automatically when using **Google Colab**.

---

## 🪪 License

This project is open-source and available under the **MIT License** —  
free to use, modify, and distribute with proper credit.


