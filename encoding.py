import pickle
import numpy as np

with open(r"C:\Users\NimaLama\OneDrive\Desktop\Project\encodings_list.pkl", "rb") as f:
    encodings = pickle.load(f)

with open(r"C:\Users\NimaLama\OneDrive\Desktop\Project\enc_names_list.pkl","rb") as r:
    label_list = pickle.load(r)
    
print("Type:", type(encodings))
print("Shape:", encodings.shape)
print("First encoding vector (first 10 values):")
print(encodings[500][:255])

print("\nMean of encodings:", np.mean(encodings))
print("Min:", np.min(encodings))
print("Max:", np.max(encodings))

norms = np.linalg.norm(encodings, axis=1)
print("Min norm:", norms.min(), "Max norm:", norms.max())
print("Zero-norm embeddings:", np.sum(norms == 0))






import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.applications.resnet import preprocess_input

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

# -------------------------------
# Load preprocessed images & labels
# -------------------------------
images = np.load(r"C:\Users\NimaLama\OneDrive\Desktop\Model_Training_and_Evaluation\images.npy")
labels = np.load(r"C:\Users\NimaLama\OneDrive\Desktop\Model_Training_and_Evaluation\labels.npy", allow_pickle=True)

print("✅ Images and labels loaded!")
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)

# -------------------------------
# Load your trained resnet model
# -------------------------------
best_model_file = r"C:\Users\NimaLama\OneDrive\Desktop\Project\best_model_resnet.h5"
model = tf.keras.models.load_model(best_model_file)
print(model.summary())

# -------------------------------
# Create encoder (cut off before softmax)
# -------------------------------
encoder = tf.keras.Model(inputs=model.input, outputs=model.get_layer('dense_1').output)

# -------------------------------
# Preprocess images for ResNet
# -------------------------------
images_preprocessed = preprocess_input(images.astype('float32'))

# -------------------------------
# Generate embeddings
# -------------------------------
batch_size = 4  # smaller batch to reduce memory usage
encodings = encoder.predict(images_preprocessed, batch_size=batch_size, verbose=1)


print("✅ Embeddings generated!")
print("Encodings shape:", encodings.shape)
print("First encoding vector (first 10 values):", encodings[0][:10])

# -------------------------------
# Save embeddings and labels
# -------------------------------
enc_path = r"C:\Users\NimaLama\OneDrive\Desktop\Project\encodings_list.pkl"
names_path = r"C:\Users\NimaLama\OneDrive\Desktop\Project\enc_names_list.pkl"

with open(enc_path, 'wb') as f:
    pickle.dump(encodings, f)

with open(names_path, 'wb') as f:
    pickle.dump(labels.tolist(), f)

print("✅ Encodings and names saved successfully!")
print(f"Encodings saved at: {enc_path}")
print(f"Names saved at: {names_path}")



