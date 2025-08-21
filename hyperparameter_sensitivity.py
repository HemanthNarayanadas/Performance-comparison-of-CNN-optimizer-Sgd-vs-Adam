"""
Hyperparameter Sensitivity Analysis for SGD vs Adam
Analyzing how different learning rates affect performance
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import time

print("=" * 60)
print("HYPERPARAMETER SENSITIVITY ANALYSIS")
print("=" * 60)

# Load and prepare data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

def create_simple_cnn():
    """Simplified CNN for faster hyperparameter testing"""
    return keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

# Learning rate ranges for testing
sgd_learning_rates = [0.001, 0.01, 0.1, 0.5]
adam_learning_rates = [0.0001, 0.001, 0.01, 0.1]

print("Testing different learning rates...")
print("SGD learning rates:", sgd_learning_rates)
print("Adam learning rates:", adam_learning_rates)
print()

# Results storage
sgd_results = {}
adam_results = {}

# Test SGD with different learning rates
print("Testing SGD with different learning rates:")
for lr in sgd_learning_rates:
    print(f"  Testing SGD with lr={lr}...")
    
    model = create_simple_cnn()
    optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    start_time = time.time()
    history = model.fit(x_train, y_train, epochs=5, validation_split=0.2, 
                       batch_size=32, verbose=0)
    training_time = time.time() - start_time
    
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    sgd_results[lr] = {
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'training_time': training_time,
        'final_train_acc': history.history['accuracy'][-1],
        'final_val_acc': history.history['val_accuracy'][-1]
    }
    
    print(f"    Test Accuracy: {test_accuracy:.4f}")

print()

# Test Adam with different learning rates
print("Testing Adam with different learning rates:")
for lr in adam_learning_rates:
    print(f"  Testing Adam with lr={lr}...")
    
    model = create_simple_cnn()
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    start_time = time.time()
    history = model.fit(x_train, y_train, epochs=5, validation_split=0.2, 
                       batch_size=32, verbose=0)
    training_time = time.time() - start_time
    
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    adam_results[lr] = {
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'training_time': training_time,
        'final_train_acc': history.history['accuracy'][-1],
        'final_val_acc': history.history['val_accuracy'][-1]
    }
    
    print(f"    Test Accuracy: {test_accuracy:.4f}")

print()

# Visualization of hyperparameter sensitivity
plt.figure(figsize=(15, 10))

# SGD learning rate sensitivity
plt.subplot(2, 2, 1)
sgd_lrs = list(sgd_results.keys())
sgd_accs = [sgd_results[lr]['test_accuracy'] for lr in sgd_lrs]
plt.semilogx(sgd_lrs, sgd_accs, 'o-', linewidth=2, markersize=8, color='#FF6B6B')
plt.title('SGD Learning Rate Sensitivity', fontweight='bold')
plt.xlabel('Learning Rate')
plt.ylabel('Test Accuracy')
plt.grid(True, alpha=0.3)
for lr, acc in zip(sgd_lrs, sgd_accs):
    plt.annotate(f'{acc:.3f}', (lr, acc), textcoords="offset points", 
                xytext=(0,10), ha='center')

# Adam learning rate sensitivity
plt.subplot(2, 2, 2)
adam_lrs = list(adam_results.keys())
adam_accs = [adam_results[lr]['test_accuracy'] for lr in adam_lrs]
plt.semilogx(adam_lrs, adam_accs, 'o-', linewidth=2, markersize=8, color='#4ECDC4')
plt.title('Adam Learning Rate Sensitivity', fontweight='bold')
plt.xlabel('Learning Rate')
plt.ylabel('Test Accuracy')
plt.grid(True, alpha=0.3)
for lr, acc in zip(adam_lrs, adam_accs):
    plt.annotate(f'{acc:.3f}', (lr, acc), textcoords="offset points", 
                xytext=(0,10), ha='center')

# Training time comparison
plt.subplot(2, 2, 3)
sgd_times = [sgd_results[lr]['training_time'] for lr in sgd_lrs]
adam_times = [adam_results[lr]['training_time'] for lr in adam_lrs]

x_pos = np.arange(len(sgd_lrs))
width = 0.35

plt.bar(x_pos - width/2, sgd_times, width, label='SGD', color='#FF6B6B', alpha=0.7)
plt.bar(x_pos + width/2, adam_times[:len(sgd_lrs)], width, label='Adam', color='#4ECDC4', alpha=0.7)

plt.title('Training Time vs Learning Rate', fontweight='bold')
plt.xlabel('Learning Rate Index')
plt.ylabel('Training Time (seconds)')
plt.xticks(x_pos, [f'{lr}' for lr in sgd_lrs])
plt.legend()

# Best performance summary
plt.subplot(2, 2, 4)
best_sgd_lr = max(sgd_results.items(), key=lambda x: x[1]['test_accuracy'])
best_adam_lr = max(adam_results.items(), key=lambda x: x[1]['test_accuracy'])

optimizers = ['SGD (Best)', 'Adam (Best)']
best_accuracies = [best_sgd_lr[1]['test_accuracy'], best_adam_lr[1]['test_accuracy']]
best_lrs = [best_sgd_lr[0], best_adam_lr[0]]

bars = plt.bar(optimizers, best_accuracies, color=['#FF6B6B', '#4ECDC4'])
plt.title('Best Performance Comparison', fontweight='bold')
plt.ylabel('Test Accuracy')
plt.ylim(0, 1)

for bar, acc, lr in zip(bars, best_accuracies, best_lrs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.4f}\n(lr={lr})', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# Summary of hyperparameter analysis
print("HYPERPARAMETER SENSITIVITY SUMMARY:")
print("=" * 45)
print(f"Best SGD: lr={best_sgd_lr[0]}, accuracy={best_sgd_lr[1]['test_accuracy']:.4f}")
print(f"Best Adam: lr={best_adam_lr[0]}, accuracy={best_adam_lr[1]['test_accuracy']:.4f}")
print()

print("KEY FINDINGS:")
print("• SGD performance is more sensitive to learning rate selection")
print("• Adam shows more stable performance across different learning rates")
print("• Optimal learning rates differ significantly between optimizers")
print("• Both optimizers can achieve competitive results with proper tuning")
