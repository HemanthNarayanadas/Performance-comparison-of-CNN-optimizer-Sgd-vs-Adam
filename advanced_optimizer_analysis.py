"""
Advanced CNN Optimizer Analysis
Extended analysis with additional optimizers and metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import time

print("=" * 60)
print("ADVANCED CNN OPTIMIZER ANALYSIS")
print("=" * 60)

# Load data (reusing from main analysis)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

def create_cnn_model():
    """Create CNN model for comparison"""
    return keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])

# Extended optimizer comparison
optimizers_extended = {
    'SGD': keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    'Adam': keras.optimizers.Adam(learning_rate=0.001),
    'RMSprop': keras.optimizers.RMSprop(learning_rate=0.001),
    'AdamW': keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01)
}

print("Extended Optimizer Comparison:")
print("Testing SGD, Adam, RMSprop, and AdamW optimizers...")
print()

results_extended = {}
EPOCHS = 10  # Reduced for demonstration

for optimizer_name, optimizer in optimizers_extended.items():
    print(f"Training with {optimizer_name}...")
    
    model = create_cnn_model()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    start_time = time.time()
    history = model.fit(x_train, y_train, epochs=EPOCHS, validation_split=0.2, 
                       batch_size=32, verbose=0)
    training_time = time.time() - start_time
    
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    results_extended[optimizer_name] = {
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'training_time': training_time,
        'history': history.history
    }
    
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    print(f"  Training Time: {training_time:.2f}s")
    print()

# Visualization of extended results
plt.figure(figsize=(15, 10))

# Test accuracy comparison
plt.subplot(2, 2, 1)
optimizers = list(results_extended.keys())
accuracies = [results_extended[opt]['test_accuracy'] for opt in optimizers]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
bars = plt.bar(optimizers, accuracies, color=colors)
plt.title('Test Accuracy Comparison (Extended)', fontweight='bold')
plt.ylabel('Test Accuracy')
plt.ylim(0, 1)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

# Training time comparison
plt.subplot(2, 2, 2)
times = [results_extended[opt]['training_time'] for opt in optimizers]
bars = plt.bar(optimizers, times, color=colors)
plt.title('Training Time Comparison (Extended)', fontweight='bold')
plt.ylabel('Training Time (seconds)')
for bar, time_val in zip(bars, times):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')

# Loss curves
plt.subplot(2, 2, 3)
for optimizer_name, result in results_extended.items():
    plt.plot(result['history']['loss'], label=optimizer_name, linewidth=2)
plt.title('Training Loss Curves', fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Accuracy curves
plt.subplot(2, 2, 4)
for optimizer_name, result in results_extended.items():
    plt.plot(result['history']['accuracy'], label=optimizer_name, linewidth=2)
plt.title('Training Accuracy Curves', fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Summary of extended analysis
print("EXTENDED ANALYSIS SUMMARY:")
print("=" * 40)
best_accuracy = max(results_extended.items(), key=lambda x: x[1]['test_accuracy'])
fastest_training = min(results_extended.items(), key=lambda x: x[1]['training_time'])

print(f"Best Accuracy: {best_accuracy[0]} ({best_accuracy[1]['test_accuracy']:.4f})")
print(f"Fastest Training: {fastest_training[0]} ({fastest_training[1]['training_time']:.2f}s)")
print()

print("OPTIMIZER CHARACTERISTICS:")
print("• SGD: Simple, memory efficient, requires careful tuning")
print("• Adam: Adaptive, generally good performance, more memory usage")
print("• RMSprop: Good for RNNs, adaptive learning rate")
print("• AdamW: Adam with weight decay, often better generalization")
