import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications.resnet import preprocess_input
import matplotlib.pyplot as plt


# GPU & Mixed Precision Setup
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

# Load Data
allImages = np.load(r"C:\Users\NimaLama\OneDrive\Desktop\Model_Training_and_Evaluation\images.npy")
allLabels = np.load(r"C:\Users\NimaLama\OneDrive\Desktop\Model_Training_and_Evaluation\labels.npy")

# Normalize images
allImagesForModel = preprocess_input(allImages)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    allImagesForModel, allLabels, test_size=0.2, random_state=42, stratify=allLabels
)

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train) #chathpy
y_test_encoded = label_encoder.transform(y_test)

# One-hot encode labels
y_train_one_hot = tf.keras.utils.to_categorical(y_train_encoded)
y_test_one_hot = tf.keras.utils.to_categorical(y_test_encoded)

# Check class weights
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train_encoded),
    y=y_train_encoded
)
class_weight_dict = dict(enumerate(class_weights))

# Model Definition (ResNet50)
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

best_model_file = r"C:\Users\NimaLama\OneDrive\Desktop\Project\best_model_resnet.h5"

callbacks = [
    ModelCheckpoint(best_model_file, verbose=1, save_best_only=True, monitor="val_accuracy"),
    ReduceLROnPlateau(monitor="val_accuracy", patience=5, factor=0.1, verbose=1, min_lr=1e-6),
    EarlyStopping(monitor="val_accuracy", patience=10, verbose=1, restore_best_weights=True)
]

# Model Training
model.compile(
    optimizer=opt,
    loss="categorical_crossentropy",
    metrics=['accuracy',tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)

hist = model.fit(
    X_train, y_train_one_hot,
    validation_data=(X_test, y_test_one_hot),
    epochs=100,
    callbacks=callbacks,
    batch_size=8,
    class_weight= class_weight_dict
)


# Evaluation & Visualization
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


# Model Performance evaluation
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

model = tf.keras.models.load_model(r"C:/Users/nitro/OneDrive/Desktop/Project/best_model_resnet.h5")

# Predict
y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_test_one_hot, axis=1)

# Class names
class_names = label_encoder.classes_
from sklearn.metrics import ( accuracy_score,
    f1_score, precision_score, recall_score
)

# Basic metrics
accuracy = accuracy_score(y_true_classes, y_pred_classes)
macro_f1 = f1_score(y_true_classes, y_pred_classes, average='macro')
weighted_f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
macro_precision = precision_score(y_true_classes, y_pred_classes, average='macro')
macro_recall = recall_score(y_true_classes, y_pred_classes, average='macro')

#  classification report
print("\n📊 Detailed Classification Report:\n")
print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

# Summary metrics overview
print("📈 Overall Metrics Summary:")
print(f"✅ Accuracy:        {accuracy:.4f}")
print(f"🎯 Macro Precision: {macro_precision:.4f}")
print(f"📡 Macro Recall:    {macro_recall:.4f}")
print(f"⚖️  Macro F1-score:  {macro_f1:.4f}")
print(f"🧮 Weighted F1:      {weighted_f1:.4f}")

# Confusion matrix heatmap
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Heatmap')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# Visual Summary Bar Chart
metrics_names = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1 (Macro)', 'Weighted F1']
metrics_values = [accuracy, macro_precision, macro_recall, macro_f1, weighted_f1]

plt.figure(figsize=(8, 6))
sns.barplot(x=metrics_names, y=metrics_values,width=0.5, palette='viridis')

plt.xticks(rotation=45, ha='right')
plt.yticks(np.linspace(0, 1, 11))
plt.ylim(0, 1)
plt.ylabel('Score', fontsize=12)

plt.title('Model Performance Summary', fontsize=14, weight='bold')

plt.tight_layout()
plt.show()
