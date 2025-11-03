import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import mixed_precision

tf.keras.backend.clear_session()
# -------------------------------
# GPU Optimization
# -------------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
              # Pre-allocate 80% of GPU memory
            tf.config.experimental.set_virtual_device_configuration(
            gpu,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6000)]  # in MB
        )
        print("Memory limit set.")
    except RuntimeError as e:
        print(e)

# Enable mixed precision for faster training & lower memory
mixed_precision.set_global_policy('mixed_float16')

# -------------------------------
# Load Data
# -------------------------------
allImages = np.load(r"C:\Users\NimaLama\OneDrive\Desktop\Model_Training_and_Evaluation\images.npy")
allLabels = np.load(r"C:\Users\NimaLama\OneDrive\Desktop\Model_Training_and_Evaluation\labels.npy")

allImagesResized = np.array([cv2.resize(img, (160, 160)) for img in allImages])
# Normalize images
allImagesForModel = allImagesResized / 255.0

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    allImagesForModel, allLabels, test_size=0.2, random_state=42, stratify=allLabels
)

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# One-hot encode labels
y_train_one_hot = tf.keras.utils.to_categorical(y_train_encoded)
y_test_one_hot = tf.keras.utils.to_categorical(y_test_encoded)

# -------------------------------
# Model Definition
# -------------------------------
batch_size = 4 # smaller batch to fit GPU memory
img_height, img_width = 160, 160
num_classes = len(np.unique(y_train))

base_model = DenseNet201(include_top=False, weights='imagenet', input_shape=(img_height, img_width, 3))

model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax', dtype='float32'))  # output in float32

opt = tf.keras.optimizers.Adam(learning_rate=1e-4)

best_model_file = r"C:\Users\NimaLama\OneDrive\Desktop\Model_Training_and_Evaluation\best_model.h5"

callbacks = [
    ModelCheckpoint(best_model_file, verbose=1, save_best_only=True, monitor="val_accuracy"),
    ReduceLROnPlateau(monitor="val_accuracy", patience=5, factor=0.1, verbose=1, min_lr=1e-6),
    EarlyStopping(monitor="val_accuracy", patience=20, verbose=1)
]

# -------------------------------
# Compile & Train
# -------------------------------
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])

hist = model.fit(
    X_train, y_train_one_hot,
    validation_data=(X_test, y_test_one_hot),
    epochs=100,
    batch_size=batch_size,
    shuffle=True,
    callbacks=callbacks
)

# -------------------------------
# Highest Validation Accuracy
# -------------------------------
highest_val_accuracy = max(hist.history["val_accuracy"])
print(f"Highest Validation Accuracy: {highest_val_accuracy}")

import matplotlib.pyplot as plt

# Plot Accuracy
plt.figure(figsize=(10,5))
plt.plot(hist.history['accuracy'], label='Training Accuracy')
plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot Loss
plt.figure(figsize=(10,5))
plt.plot(hist.history['loss'], label='Training Loss')
plt.plot(hist.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

