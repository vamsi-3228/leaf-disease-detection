# ================================
# PHASE 3: IMAGE PREPROCESSING
# ================================

# Step 1: Import libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Step 2: Define constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# Step 3: Training data generator (with augmentation)
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,        # Normalize pixels
    rotation_range=30,        # Rotate images
    zoom_range=0.2,           # Zoom images
    shear_range=0.2,          # Shear transformation
    horizontal_flip=True,     # Flip images
    fill_mode='nearest'
)

# Step 4: Validation data generator (NO augmentation)
val_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

# Step 5: Load training images
train_generator = train_datagen.flow_from_directory(
    directory="dataset/train",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# Step 6: Load validation images
val_generator = val_datagen.flow_from_directory(
    directory="dataset/val",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# Step 7: Confirmation output
print("\nImage preprocessing completed successfully!")
print("Detected classes:")
print(train_generator.class_indices)