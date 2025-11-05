import os
import tensorflow as tf
import numpy as np
import pickle
import json
import re
from tensorflow.keras.applications.resnet50 import preprocess_input


# Load encoder (remove final classification layer)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'data', 'best_model_resnet.h5')
model = tf.keras.models.load_model(MODEL_PATH)
encoder = tf.keras.Model(inputs=model.input, outputs=model.get_layer('dense_1').output)

# Load stored embeddings and names
with open(os.path.join(os.path.dirname(__file__), 'data', 'encodings_list.pkl'), 'rb') as f:
    enc_list = pickle.load(f)

with open(os.path.join(os.path.dirname(__file__), 'data', 'enc_names_list.pkl'), 'rb') as f:
    names_list = pickle.load(f)

# Load recipes data
with open(os.path.join(os.path.dirname(__file__), 'data', 'recipes.json'), 'r', encoding='utf-8') as f:
    recipes_data = json.load(f)

recipes_by_category = {entry["category"].lower(): entry["recipes"] for entry in recipes_data}

# === Helper Functions ===
# compute cosine similarity
def euclidean_norm(vec):
    return sum(x**2 for x in vec) ** 0.5

def cosine_similarity(vec1, vec2, eps=1e-10):
    dot_product = sum(x * y for x, y in zip(vec1, vec2))
    norm1 = euclidean_norm(vec1)
    norm2 = euclidean_norm(vec2)
    return dot_product / max(norm1 * norm2, eps)

# get image encoding
def get_encodings(img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.preprocessing.image.smart_resize(img_array, size=(224, 224))
    img_array = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_array)
    encoding = encoder.predict(img_preprocessed).astype('float32')
    return encoding.flatten()

# recipe recommender

def get_recipes(img, top_k=1, num_recipes=3,similarity_threshold=0.6):
    enc = get_encodings(img)
    
    # ensure enc is a numpy array
    if isinstance(enc, np.ndarray) and enc.ndim > 1:
        enc = enc.flatten()

    similarity_list = [
        float(cosine_similarity(stored_enc, enc))  # ensure it's JSON serializable
        for stored_enc in enc_list
    ]
    max_sim = max(similarity_list)
    if max_sim < similarity_threshold:
        # Image is too different from known data
        return [{"category": "unknown", "similarity": max_sim, "recipes": []}]
    
    sorted_list = sorted(zip(similarity_list, names_list), reverse=True)

    seen_categories = set()
    recommendations = []

    for sim, name in sorted_list:
        category_key = re.sub(r'[0-9]+.jpg', "", str(name)).strip().lower()
        if category_key not in seen_categories:
            seen_categories.add(category_key)
            if category_key in recipes_by_category:
                recipes = recipes_by_category[category_key][:num_recipes]

                # Ensure recipes are dicts, not objects or numpy types
                clean_recipes = []
                for r in recipes:
                    if isinstance(r, dict):
                        clean_recipes.append(r)
                    else:
                        clean_recipes.append({"name": str(r)})

                recommendations.append({
                    "category": category_key,
                    "similarity": round(float(sim), 3),
                    "recipes": clean_recipes
                })
            if len(recommendations) >= top_k:
                break

    return recommendations

