from google.colab import files
uploaded = files.upload()
import zipfile, os
zip_path = "/content/archive (2).zip"  # your zip filename
extract_path = "dataset"
if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("✅ Dataset extracted to:", extract_path)
else:
    print("✅ Dataset already exists:", extract_path)
  # =====================================================================
# CNN OPTIMIZER PERFORMANCE COMPARISON: SGD vs ADAM
# Universal Dataset Version (Works with Any Image Dataset)
# =====================================================================

import os, time, random, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

warnings.filterwarnings("ignore")

# ==========================================================
# STEP 0: CONFIGURATION & SETUP
# ==========================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

EPOCHS = 5
BATCH_SIZE = 32
IMG_SIZE = (64, 64)
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

print("=" * 75)
print(" UNIVERSAL CNN OPTIMIZER PERFORMANCE COMPARISON: SGD vs ADAM ")
print("=" * 75)
print()

# ==========================================================
# STEP 1: DATA LOADING
# ==========================================================
print("STEP 1: DATA LOADING")
print("-" * 40)
DATA_DIR = Path("dataset") # Assuming 'dataset' is the root of your extracted data

if not DATA_DIR.exists():
    print("📥 No dataset found — downloading CIFAR-10 and preparing folder structure...")
    (train_images, train_labels), (_, _) = tf.keras.datasets.cifar10.load_data()
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    for cname in class_names:
        os.makedirs(DATA_DIR / cname, exist_ok=True)

    for img, label in zip(train_images, train_labels):
        cname = class_names[int(label)]
        count = len(os.listdir(DATA_DIR / cname))
        path = DATA_DIR / cname / f"{count}.jpg"
        tf.keras.preprocessing.image.save_img(str(path), img)
    print("✅ CIFAR-10 dataset prepared at:", DATA_DIR)
else:
    print("✅ Using existing dataset:", DATA_DIR)
    # Assuming class directories are directly under DATA_DIR
    class_names = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
    # Filter out 'test', 'train', 'valid' if they are present as class names
    class_names = [c for c in class_names if c not in ['test', 'train', 'valid']]

    print(f"📂 Found {len(class_names)} classes:")
    print(class_names)

    # Display sample images from each class
    print("\n🖼️ Displaying sample images from classes...")
    plt.figure(figsize=(12, 8))
    displayed_count = 0
    for i, cname in enumerate(class_names):
        if displayed_count >= 9:
            break
        class_dir = DATA_DIR / cname
        # Find the first image file in the directory
        img_path = None
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            found_files = list(class_dir.glob(ext))
            if found_files:
                img_path = found_files[0]
                break

        if img_path and img_path.is_file():
            try:
                img = image.load_img(img_path, target_size=IMG_SIZE)
                plt.subplot(3, 3, displayed_count + 1)
                plt.imshow(img)
                plt.title(cname)
                plt.axis("off")
                displayed_count += 1
            except Exception as e:
                print(f"Could not load image {img_path}: {e}")
        else:
            print(f"No image files found for class: {cname}")

    if displayed_count > 0:
        plt.tight_layout()
        plt.show()
    else:
        print("No sample images could be displayed.")


print()

# ==========================================================
# STEP 2: DATA PREPROCESSING
# ==========================================================
print("STEP 2: DATA PREPROCESSING")
print("-" * 40)

def load_and_preprocess_data():
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    train_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='training',
        seed=SEED
    )
    val_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='validation',
        seed=SEED
    )
    print(f"✅ Data loaded: {train_gen.samples} training images, {val_gen.samples} validation images\n")
    return train_gen, val_gen

train_gen, val_gen = load_and_preprocess_data()

# ==========================================================
# STEP 3: MODEL DEFINITION
# ==========================================================
print("STEP 3: DEFINING CNN MODEL")
print("-" * 40)

def create_cnn(input_shape=(64, 64, 3), num_classes=None, wd=1e-4):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(wd), input_shape=input_shape),
        layers.BatchNormalization(), layers.ReLU(),
        layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(wd)),
        layers.BatchNormalization(), layers.ReLU(),
        layers.MaxPooling2D(), layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(wd)),
        layers.BatchNormalization(), layers.ReLU(),
        layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(wd)),
        layers.BatchNormalization(), layers.ReLU(),
        layers.MaxPooling2D(), layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(256, kernel_regularizer=regularizers.l2(wd)),
        layers.BatchNormalization(), layers.ReLU(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

num_classes = train_gen.num_classes
model_example = create_cnn(num_classes=num_classes)
model_example.summary()
print()

# ==========================================================
# STEP 4: TRAINING WITH DIFFERENT OPTIMIZERS
# ==========================================================
print("STEP 4: TRAINING WITH SGD AND ADAM OPTIMIZERS")
print("-" * 40)

optimizers = {
    "SGD": tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    "Adam": tf.keras.optimizers.Adam(learning_rate=0.001)
}

results, histories = {}, {}

def train_model(optimizer_name, optimizer, train_gen, val_gen):
    model = create_cnn(input_shape=(64, 64, 3), num_classes=num_classes)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print(f"\n🚀 Training using {optimizer_name} optimizer...")
    start = time.time()
    history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, verbose=2)
    end = time.time()

    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    print(f"✅ {optimizer_name} | Val Accuracy: {val_acc:.4f} | Time: {end - start:.2f}s\n")

    return history.history, val_acc, end - start

for name, opt in optimizers.items():
    hist, acc, elapsed = train_model(name, opt, train_gen, val_gen)
    results[name] = {"Validation Accuracy": acc, "Training Time (s)": elapsed}
    histories[name] = hist

# ==========================================================
# STEP 5: VISUALIZATION
# ==========================================================
print("STEP 5: VISUALIZING TRAINING PERFORMANCE")
print("-" * 40)

plt.figure(figsize=(10,5))
for opt in histories:
    plt.plot(histories[opt]['val_accuracy'], label=f"{opt} Val Accuracy")
plt.title("Validation Accuracy Comparison (SGD vs Adam)")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(RESULTS_DIR / "accuracy_comparison.png")
plt.show()

plt.figure(figsize=(10,5))
for opt in histories:
    plt.plot(histories[opt]['val_loss'], label=f"{opt} Val Loss")
plt.title("Validation Loss Comparison (SGD vs Adam)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(RESULTS_DIR / "loss_comparison.png")
plt.show()

# ==========================================================
# STEP 6: FINAL PERFORMANCE SUMMARY
# ==========================================================
print("STEP 6: PERFORMANCE SUMMARY")
print("-" * 40)

summary_df = pd.DataFrame(results).T
print(summary_df)
summary_df.to_csv(RESULTS_DIR / "optimizer_results.csv", index=True)

best_opt = summary_df["Validation Accuracy"].idxmax()
print(f"\n🏆 Best Optimizer: {best_opt} with Accuracy = {summary_df.loc[best_opt, 'Validation Accuracy']:.4f}")
print("\n✅ Experiment Completed Successfully!")
print(f"Results and plots saved in: {RESULTS_DIR.resolve()}")
