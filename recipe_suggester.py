import os
import tensorflow as tf
import numpy as np
import pickle
import re
from tensorflow.keras.applications.resnet import preprocess_input

#  Load your trained model
best_model_file = r"C:\Users\nitro\OneDrive\Desktop\Project\best_model_resnet.h5"
model = tf.keras.models.load_model(best_model_file)

# Cut off before the final classification layer (encoder)
encoder = tf.keras.Model(inputs=model.input, outputs=model.get_layer('dense_1').output)

# 🔹 Load saved encodings
enc_path = r"C:\Users\NimaLama\OneDrive\Desktop\Project\encodings_list.pkl"
names_path = r"C:\Users\NimaLama\OneDrive\Desktop\Project\enc_names_list.pkl"

with open(enc_path, 'rb') as f:
    enc_list = pickle.load(f)
    enc_list = np.array(enc_list)

with open(names_path, 'rb') as f:
    names_list = pickle.load(f)

print("✅ Encodings and names loaded successfully!")
print("Encodings shape:", enc_list.shape)
print("Number of labels:", len(names_list))

# Helper Functions
def get_encodings(img):
    """
    Preprocess an image and extract its encoding using your trained ResNet encoder.
    Args:
        img: A PIL image object.
    Returns:
        A 1D NumPy array encoding.
    """
    # Convert PIL image → array
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    # Resize to model input shape (change if your model trained on a different size)
    img_array = tf.keras.preprocessing.image.smart_resize(img_array, size=(224, 224))

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess using ResNet preprocess function
    img_preprocessed = preprocess_input(img_array)

    # Extract encoding
    encoding = encoder.predict(img_preprocessed).astype('float32')
    # Flatten
    encoding = encoding.flatten()
    
    return encoding

import json

recipes_file = r"C:\Users\NimaLama\OneDrive\Desktop\Project\recipes.json"

with open(recipes_file, 'r', encoding='utf-8') as f:
    recipes_data = json.load(f)

# Convert JSON into a dictionary for fast lookup
recipes_by_category = {entry["category"].lower(): entry["recipes"] for entry in recipes_data}

print(f"✅ Loaded {len(recipes_by_category)} categories from JSON!")


# compute euclidean distance(norm) of vectors
def euclidean_norm(vec):
    return sum(x**2 for x in vec) ** 0.5

# compute cosine similarity between two vectors
def cosine_similarity(vec1, vec2, eps=1e-10):
    # Flatten vectors if needed
    vec1 = list(vec1)
    vec2 = list(vec2)
    
    dot_product = sum(x * y for x, y in zip(vec1, vec2))
    norm1 = euclidean_norm(vec1)
    norm2 = euclidean_norm(vec2)

    return dot_product / max(norm1 * norm2, eps)

def get_recipes(img, top_k=3, num_recipes=10):
    """
    Given an image, returns top matching categories and corresponding recipes.

    Args:
        img: PIL image object
        top_k: Number of top categories to return
        num_recipes: Number of recipes to show per category

    Returns:
        List of dictionaries containing category, similarity, and top recipes
    """
    enc = get_encodings(img)  # Should return a 1D or flattened embedding
    similarity_list = []
    
    # Compute cosine similarity with all stored embeddings
    for stored_enc in enc_list:
        sim = cosine_similarity(stored_enc, enc)
        similarity_list.append(float(sim))

    # Sort embeddings by similarity descending
    sorted_list = sorted(zip(similarity_list, names_list), reverse=True)

    seen_categories = set()
    recommendations = []

    for sim, name in sorted_list:
        # Extract category from filename
        category_key = re.sub(r'[0-9]+.jpg', "", str(name)).strip().lower()
        if category_key not in seen_categories:
            seen_categories.add(category_key)
            if category_key in recipes_by_category:
                recipes = recipes_by_category[category_key][:num_recipes]
                recommendations.append({
                    "category": category_key,
                    "similarity": round(sim, 3),
                    "recipes": recipes
                })
            if len(recommendations) >= top_k:
                break

    return recommendations

# Example usage
if __name__ == "__main__":
    from PIL import Image

    test_image_path = r"C:\Users\NimaLama\Downloads\download (4).jpg"
    img = Image.open(test_image_path).convert('RGB')

    results = get_recipes(img, top_k=1, num_recipes=3)

    print("\n🍽️ Recommended Recipes:")
    for result in results:
        print(f"\n📸 Category: {result['category'].capitalize()} (similarity: {result['similarity']})")
        for i, recipe in enumerate(result['recipes'], 1):
            print(f"  {i}. {recipe['name']} — ⏱ {recipe['cooking_time']}")
            print(f"     🧂 Ingredients: {recipe['ingredients']}")
            print(f"     👨‍🍳 Directions: {recipe['directions'][:150]}...")

