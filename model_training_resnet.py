import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt

# -------------------------------
# GPU & Mixed Precision Setup
# -------------------------------
tf.keras.backend.clear_session()

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6000)]
            )
        print("GPU memory limit set to 6GB.")
    except RuntimeError as e:
        print(e)

mixed_precision.set_global_policy('mixed_float16')

# # -------------------------------
# Load Data
# -------------------------------
allImages = np.load(r"C:\Users\NimaLama\OneDrive\Desktop\Model_Training_and_Evaluation\images.npy")
allLabels = np.load(r"C:\Users\NimaLama\OneDrive\Desktop\Model_Training_and_Evaluation\labels.npy")

# Normalize images
allImagesForModel = allImages / 255.0

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

# Check if you need class weights
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train_encoded),
    y=y_train_encoded
)
class_weight_dict = dict(enumerate(class_weights))
# -------------------------------
# Model Definition (ResNet50)
# -------------------------------
img_height, img_width = 224, 224
num_classes = len(np.unique(y_train))

base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(img_height, img_width, 3))

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5), 
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3), 
    layers.Dense(num_classes, activation='softmax')
])

opt = tf.keras.optimizers.Adam(learning_rate=1e-4)

best_model_file = r"C:\Users\NimaLama\OneDrive\Desktop\Model_Training_and_Evaluation\best_model_resnet.h5"

callbacks = [
    ModelCheckpoint(best_model_file, verbose=1, save_best_only=True, monitor="val_accuracy"),
    ReduceLROnPlateau(monitor="val_accuracy", patience=5, factor=0.1, verbose=1, min_lr=1e-6),
    EarlyStopping(monitor="val_accuracy", patience=10, verbose=1, restore_best_weights=True)
]

# -------------------------------
# Initial Training (Frozen base)
# -------------------------------
model.compile(
    optimizer=opt,
    loss="categorical_crossentropy",
    metrics=['accuracy',tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)

# print("\n🔹 Starting initial training (frozen base)...")
# hist1 = model.fit(
#     X_train, y_train_one_hot,
#     validation_data=(X_test, y_test_one_hot),
#     epochs=20,
#     callbacks=callbacks,
#     class_weight=class_weight_dict
# )

# # -------------------------------
# # Fine-Tuning (Unfreeze top layers)
# # -------------------------------
# print("\n🔹 Fine-tuning top layers of ResNet50...")

# # Unfreeze top few layers of ResNet50
# base_model.trainable = True
# for layer in base_model.layers[:-50]:  # Freeze first N layers, fine-tune last 50
#     layer.trainable = False

# # Compile with a smaller learning rate for fine-tuning
# fine_tune_lr = 1e-5
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
#     loss="categorical_crossentropy",
#     metrics=['accuracy',f1_score]
# )

hist = model.fit(
    X_train, y_train_one_hot,
    validation_data=(X_test, y_test_one_hot),
    epochs=100,
    callbacks=callbacks,
    batch_size=8,
    class_weight= class_weight_dict
)

# -------------------------------
# # Combine training histories
# # -------------------------------
# final_history = {
#     key: hist1.history[key] + hist2.history[key]
#     for key in hist1.history.keys()
#     if key in hist2.history
# }

# -------------------------------
# Evaluation & Visualization
# -------------------------------
highest_val_acc = max(hist.history["val_accuracy"])
print(f"\n✅ Highest Validation Accuracy after Fine-Tuning: {highest_val_acc:.4f}")

plt.figure(figsize=(10,5))
plt.plot(hist.history['accuracy'], label='Training Accuracy')
plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(hist.history['loss'], label='Training Loss')
plt.plot(hist.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()



