# test_model.py

import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# -----------------------------
# 1️⃣ Check if model and image exist
# -----------------------------
MODEL_PATH = "model/leaf_disease_model.h5"
IMAGE_PATH = "test_images/leaf.jpg"

if not os.path.exists(MODEL_PATH):
    print("❌ Model file not found!")
    exit()

if not os.path.exists(IMAGE_PATH):
    print("❌ Test image not found!")
    exit()

# -----------------------------
# 2️⃣ Load trained model
# -----------------------------
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# -----------------------------
# 3️⃣ Preprocess image (STEP 4)
# -----------------------------
img = image.load_img(IMAGE_PATH, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# -----------------------------
# 4️⃣ Predict disease (STEP 5)
# -----------------------------
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)
confidence = np.max(prediction) * 100

# -----------------------------
# 5️⃣ Class labels (same order as training)
# -----------------------------
class_names = [
    "Potato_healthy",
    "Potato_Late_blight",
    "Potato_Early_blight",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_healthy",
]

disease_name = class_names[predicted_class]

# -----------------------------
# 6️⃣ Show results
# -----------------------------
print("\n🌱 Leaf Disease Detection Result")
print("--------------------------------")
print("Predicted Disease:", disease_name)
print("Confidence: {:.2f}%".format(confidence))

# Optional low-confidence warning
if confidence < 50:
    print("⚠ Low confidence prediction. Please use a clearer leaf image.")