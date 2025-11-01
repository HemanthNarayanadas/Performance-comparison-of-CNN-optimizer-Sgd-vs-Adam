🧠 CNN Optimizer Performance Comparison: SGD vs Adam

A comprehensive analysis comparing SGD (Stochastic Gradient Descent) and Adam (Adaptive Moment Estimation) optimizers on Convolutional Neural Networks (CNNs) for image classification.
This implementation supports any dataset — plug in your own image data, and the model automatically adapts.

📘 Project Overview

This project performs a detailed performance comparison between SGD and Adam optimizers using a unified CNN model.
It measures and visualizes the differences in:

Training and validation accuracy

Loss convergence

Speed and computational efficiency

Overall model stability

🏗️ Project Structure
---
cnn_optimizer_project/
├── scripts/
│   ├── cnn_optimizer_comparison.py     # Main CNN optimizer comparison code
│   └── data_loader.py                  # Dataset loading and preprocessing
├── results/
│   ├── accuracy_loss_plots.png         # Accuracy/loss plots saved automatically
│   └── summary_results.csv             # CSV summary of optimizer performance
└── README.md                           # Project documentation

⚙️ Key Features

✅ Dataset Flexibility – Works with any dataset (custom or built-in)
✅ Dual Optimizer Evaluation – CNN trained with both SGD and Adam
✅ Automatic Performance Metrics – Accuracy, loss, and training time comparison
✅ Beautiful Visualizations – Accuracy/loss curves and comparison charts
✅ Customizable Parameters – Easily modify epochs, learning rate, or model depth

🚀 How to Run
🔹 1. Clone or Download
git clone https://github.com/yourusername/cnn-optimizer-comparison.git
cd cnn-optimizer-comparison

🔹 2. Install Requirements
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn

🔹 3. Run the Comparison
python cnn_optimizer_comparison.py

📊 Key Findings
Metric	SGD	Adam	Conclusion
Accuracy	Moderate	Higher	Adam performs better overall
Convergence	Slower	Faster	Adam converges quickly
Stability	Requires tuning	More stable	Adam is smoother
Memory Usage	Low	Slightly higher	SGD is lighter
🧩 Model & Training Configuration

Architecture:
Custom CNN with Conv2D → BatchNorm → MaxPooling → Dropout → Dense → Softmax

Input: Automatically resized images

Epochs: 5 (can be increased)

Batch Size: 32

Loss Function: Categorical Crossentropy

Metrics: Accuracy

⚙️ Optimizer Parameters

SGD: learning_rate=0.01, momentum=0.9

Adam: learning_rate=0.001

📈 Visualizations

🟩 Training vs Validation Accuracy/Loss Curves
📊 Optimizer Comparison Bar Charts
🧾 Final Summary Table (Accuracy, Loss, Time)

(Outputs are automatically saved in /results folder)

🎓 Educational Value

This project helps you learn:

The difference in behavior between SGD and Adam

How optimizer choice affects CNN training and convergence

How to build a comparative ML experiment

How to visualize and interpret model results

🔮 Future Enhancements

✨ Add more optimizers (RMSprop, AdamW, AdaGrad)
✨ Implement learning rate schedulers
✨ Add transfer learning for advanced comparison
✨ Extend to larger datasets (ImageNet, TinyImageNet, etc.)
✨ Automate hyperparameter sensitivity testing

🧰 Dependencies
TensorFlow >= 2.x  
NumPy  
Pandas  
Matplotlib  
Seaborn  
Scikit-learn


Install all dependencies:

pip install -r requirements.txt

📄 License

This project is open-source and available under the MIT License.


