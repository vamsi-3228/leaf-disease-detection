import numpy as np
from tensorflow.keras.preprocessing import image

# Load image
img = image.load_img(
    "test_images/leaf.jpg",
    target_size=(224, 224)
)

# Convert image to array
img_array = image.img_to_array(img)

# Normalize pixel values
img_array = img_array / 255.0

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

print("✅ Image is preprocessed and ready for prediction")