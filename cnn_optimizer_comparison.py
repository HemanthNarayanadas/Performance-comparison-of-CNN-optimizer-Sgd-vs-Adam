"""
CNN Optimizer Performance Comparison: SGD vs Adam
A comprehensive analysis comparing SGD and Adam optimizers on CNN models
Following similar methodology to the weather forecasting LSTM project
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import time
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 60)
print("CNN OPTIMIZER PERFORMANCE COMPARISON: SGD vs ADAM")
print("=" * 60)
print()

# Step 1: Data Loading and Exploration
print("STEP 1: DATA LOADING AND EXPLORATION")
print("-" * 40)

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Class names for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Training data shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Test labels shape: {y_test.shape}")
print(f"Number of classes: {len(class_names)}")
print(f"Class names: {class_names}")
print()

# Display sample images
plt.figure(figsize=(12, 8))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(x_train[i])
    plt.title(f'{class_names[y_train[i][0]]}')
    plt.axis('off')
plt.suptitle('Sample Images from CIFAR-10 Dataset', fontsize=16)
plt.tight_layout()
plt.show()

# Step 2: Data Preprocessing
print("STEP 2: DATA PREPROCESSING")
print("-" * 40)

# Normalize pixel values to [0, 1]
x_train_normalized = x_train.astype('float32') / 255.0
x_test_normalized = x_test.astype('float32') / 255.0

# Convert labels to categorical
y_train_categorical = to_categorical(y_train, 10)
y_test_categorical = to_categorical(y_test, 10)

print("Data preprocessing completed:")
print(f"Training data range: [{x_train_normalized.min():.2f}, {x_train_normalized.max():.2f}]")
print(f"Training labels shape after categorical encoding: {y_train_categorical.shape}")
print()

# Step 3: CNN Model Architecture Definition
print("STEP 3: CNN MODEL ARCHITECTURE DEFINITION")
print("-" * 40)

def create_cnn_model():
    """
    Create a CNN model for image classification
    Similar structure to the LSTM model creation in the reference project
    """
    model = keras.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # Dense Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    return model

# Create and display model architecture
sample_model = create_cnn_model()
print("CNN Model Architecture:")
sample_model.summary()
print()

# Step 4: Training with Different Optimizers
print("STEP 4: TRAINING WITH DIFFERENT OPTIMIZERS")
print("-" * 40)

# Training parameters
EPOCHS = 20
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

# Dictionary to store results
results = {}
training_histories = {}

# Define optimizers to compare
optimizers_config = {
    'SGD': keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    'Adam': keras.optimizers.Adam(learning_rate=0.001)
}

print("Starting training with different optimizers...")
print()

for optimizer_name, optimizer in optimizers_config.items():
    print(f"Training with {optimizer_name} optimizer...")
    print("-" * 30)
    
    # Create fresh model for each optimizer
    model = create_cnn_model()
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Record training start time
    start_time = time.time()
    
    # Train model
    history = model.fit(
        x_train_normalized, y_train_categorical,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        verbose=1,
        shuffle=True
    )
    
    # Record training end time
    end_time = time.time()
    training_time = end_time - start_time
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(x_test_normalized, y_test_categorical, verbose=0)
    
    # Store results
    results[optimizer_name] = {
        'model': model,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'training_time': training_time,
        'final_train_loss': history.history['loss'][-1],
        'final_train_accuracy': history.history['accuracy'][-1],
        'final_val_loss': history.history['val_loss'][-1],
        'final_val_accuracy': history.history['val_accuracy'][-1]
    }
    
    training_histories[optimizer_name] = history.history
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print()

# Step 5: Performance Analysis and Visualization
print("STEP 5: PERFORMANCE ANALYSIS AND VISUALIZATION")
print("-" * 40)

# Create comprehensive comparison plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Training Loss Comparison
axes[0, 0].set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
for optimizer_name, history in training_histories.items():
    axes[0, 0].plot(history['loss'], label=f'{optimizer_name}', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Validation Loss Comparison
axes[0, 1].set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
for optimizer_name, history in training_histories.items():
    axes[0, 1].plot(history['val_loss'], label=f'{optimizer_name}', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Validation Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Training Accuracy Comparison
axes[0, 2].set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
for optimizer_name, history in training_histories.items():
    axes[0, 2].plot(history['accuracy'], label=f'{optimizer_name}', linewidth=2)
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Accuracy')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Validation Accuracy Comparison
axes[1, 0].set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
for optimizer_name, history in training_histories.items():
    axes[1, 0].plot(history['val_accuracy'], label=f'{optimizer_name}', linewidth=2)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Validation Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: Test Accuracy Comparison (Bar Chart)
axes[1, 1].set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
optimizers = list(results.keys())
test_accuracies = [results[opt]['test_accuracy'] for opt in optimizers]
bars = axes[1, 1].bar(optimizers, test_accuracies, color=['#FF6B6B', '#4ECDC4'])
axes[1, 1].set_ylabel('Test Accuracy')
axes[1, 1].set_ylim(0, 1)
# Add value labels on bars
for bar, acc in zip(bars, test_accuracies):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')

# Plot 6: Training Time Comparison (Bar Chart)
axes[1, 2].set_title('Training Time Comparison', fontsize=14, fontweight='bold')
training_times = [results[opt]['training_time'] for opt in optimizers]
bars = axes[1, 2].bar(optimizers, training_times, color=['#FF6B6B', '#4ECDC4'])
axes[1, 2].set_ylabel('Training Time (seconds)')
# Add value labels on bars
for bar, time_val in zip(bars, training_times):
    axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                    f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# Step 6: Detailed Performance Metrics
print("STEP 6: DETAILED PERFORMANCE METRICS")
print("-" * 40)

# Create performance comparison table
performance_data = []
for optimizer_name, result in results.items():
    performance_data.append({
        'Optimizer': optimizer_name,
        'Test Accuracy': f"{result['test_accuracy']:.4f}",
        'Test Loss': f"{result['test_loss']:.4f}",
        'Training Time (s)': f"{result['training_time']:.2f}",
        'Final Train Accuracy': f"{result['final_train_accuracy']:.4f}",
        'Final Val Accuracy': f"{result['final_val_accuracy']:.4f}",
        'Convergence Speed': 'Fast' if result['training_time'] < np.mean([r['training_time'] for r in results.values()]) else 'Slow'
    })

performance_df = pd.DataFrame(performance_data)
print("Performance Comparison Summary:")
print("=" * 80)
print(performance_df.to_string(index=False))
print()

# Step 7: Statistical Analysis
print("STEP 7: STATISTICAL ANALYSIS")
print("-" * 40)

# Calculate performance differences
sgd_accuracy = results['SGD']['test_accuracy']
adam_accuracy = results['Adam']['test_accuracy']
accuracy_difference = adam_accuracy - sgd_accuracy

sgd_time = results['SGD']['training_time']
adam_time = results['Adam']['training_time']
time_difference = adam_time - sgd_time

print("Statistical Analysis Results:")
print(f"• Accuracy Difference (Adam - SGD): {accuracy_difference:.4f}")
print(f"• Time Difference (Adam - SGD): {time_difference:.2f} seconds")
print(f"• Best Performing Optimizer: {'Adam' if adam_accuracy > sgd_accuracy else 'SGD'}")
print(f"• Faster Optimizer: {'Adam' if adam_time < sgd_time else 'SGD'}")
print()

# Step 8: Confusion Matrix Analysis
print("STEP 8: CONFUSION MATRIX ANALYSIS")
print("-" * 40)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for idx, (optimizer_name, result) in enumerate(results.items()):
    # Generate predictions
    y_pred = result['model'].predict(x_test_normalized, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test_categorical, axis=1)
    
    # Create confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[idx])
    axes[idx].set_title(f'Confusion Matrix - {optimizer_name}', fontsize=14, fontweight='bold')
    axes[idx].set_xlabel('Predicted Label')
    axes[idx].set_ylabel('True Label')

plt.tight_layout()
plt.show()

# Step 9: Learning Curve Analysis
print("STEP 9: LEARNING CURVE ANALYSIS")
print("-" * 40)

# Analyze learning curves for convergence patterns
plt.figure(figsize=(15, 10))

# Create subplots for detailed learning curve analysis
plt.subplot(2, 2, 1)
for optimizer_name, history in training_histories.items():
    plt.plot(history['loss'], label=f'{optimizer_name} Train', linewidth=2)
    plt.plot(history['val_loss'], label=f'{optimizer_name} Val', linestyle='--', linewidth=2)
plt.title('Loss Curves Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
for optimizer_name, history in training_histories.items():
    plt.plot(history['accuracy'], label=f'{optimizer_name} Train', linewidth=2)
    plt.plot(history['val_accuracy'], label=f'{optimizer_name} Val', linestyle='--', linewidth=2)
plt.title('Accuracy Curves Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Learning rate analysis (approximate)
plt.subplot(2, 2, 3)
for optimizer_name, history in training_histories.items():
    loss_diff = np.diff(history['loss'])
    plt.plot(loss_diff, label=f'{optimizer_name}', linewidth=2)
plt.title('Loss Change Rate (Learning Speed)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss Change')
plt.legend()
plt.grid(True, alpha=0.3)

# Overfitting analysis
plt.subplot(2, 2, 4)
for optimizer_name, history in training_histories.items():
    train_acc = np.array(history['accuracy'])
    val_acc = np.array(history['val_accuracy'])
    overfitting_gap = train_acc - val_acc
    plt.plot(overfitting_gap, label=f'{optimizer_name}', linewidth=2)
plt.title('Overfitting Analysis (Train - Val Accuracy)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy Gap')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Step 10: Final Conclusions and Recommendations
print("STEP 10: FINAL CONCLUSIONS AND RECOMMENDATIONS")
print("-" * 40)

print("COMPREHENSIVE ANALYSIS RESULTS:")
print("=" * 50)

# Determine winner in different categories
categories = {
    'Test Accuracy': 'Adam' if results['Adam']['test_accuracy'] > results['SGD']['test_accuracy'] else 'SGD',
    'Training Speed': 'Adam' if results['Adam']['training_time'] < results['SGD']['training_time'] else 'SGD',
    'Convergence Stability': 'Adam',  # Generally more stable
    'Memory Efficiency': 'SGD',  # Uses less memory
}

for category, winner in categories.items():
    print(f"• {category}: {winner}")

print()
print("DETAILED INSIGHTS:")
print("-" * 20)

if results['Adam']['test_accuracy'] > results['SGD']['test_accuracy']:
    print("✓ Adam achieved higher test accuracy, showing better optimization capability")
else:
    print("✓ SGD achieved higher test accuracy, demonstrating effective momentum-based learning")

if results['Adam']['training_time'] < results['SGD']['training_time']:
    print("✓ Adam converged faster, requiring less training time")
else:
    print("✓ SGD trained faster, showing computational efficiency")

print("✓ Adam typically shows more stable convergence with adaptive learning rates")
print("✓ SGD with momentum can achieve competitive results with proper tuning")

print()
print("RECOMMENDATIONS:")
print("-" * 15)
print("• For quick prototyping and general use: Adam optimizer")
print("• For production with careful hyperparameter tuning: SGD with momentum")
print("• For limited computational resources: SGD")
print("• For complex architectures: Adam or AdamW")

print()
print("=" * 60)
print("ANALYSIS COMPLETED SUCCESSFULLY!")
print("=" * 60)
