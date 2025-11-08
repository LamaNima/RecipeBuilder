# 🍽️ RecipeBuilder — AI-Powered Recipe Generator from Food Images

**RecipeBuilder** is a Django-based web application that intelligently recognizes food images and recommends corresponding recipes using deep learning. The project combines **computer vision** and **content-based recommendation** to identify dishes and retrieve recipes by comparing learned **image embeddings**.

## 🚀 Key Features

- 🧠 **CNN-based Food Recognition** — Trained using **ResNet50** as a base feature extractor to generate meaningful food embeddings.
- 🍲 **Recipe Generation via Similarity Matching** — Suggests recipes by comparing embeddings between uploaded images and stored encodings.
- 🖥️ **Interactive Web Interface** — Built with **Django**, featuring user-friendly upload and results pages.

- ## 🛠️ Tech Stack

| Component | Technology |
|------------|-------------|
| **Frontend** | HTML, CSS, JavaScript (Vanilla JS) |
| **Backend** | Django (Python) |
| **Deep Learning** | TensorFlow / Keras (ResNet50) |
| **Model Storage** | `.h5` for trained CNN model, `.pkl` for encodings |

## 📈 Workflow
1. **User uploads a food image** through the web interface.
2. The image is **preprocessed** (resized, normalized).
3. The model **extracts embeddings** using the trained ResNet backbone.
4. These embeddings are **compared with stored encodings** (from dataset images).
5. The **most similar recipes** are retrieved and displayed with details such as:
   - Recipe name
   - Ingredients
   - Cooking time


## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/RecipeBuilder.git
cd RecipeBuilder
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate       # On Linux/Mac
venv\Scripts\activate        # On Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```


### 4. Run the Django server
```bash
python manage.py runserver
```

### 6. Access the app
Open your browser and go to:
👉 `http://127.0.0.1:8000/`

---
