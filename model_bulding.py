# ================================
# PHASE 4: MODEL BUILDING
# ================================

# Step 1: Import required libraries
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model

# Step 2: Define image shape and number of classes
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = 6

# Step 3: Load MobileNetV2 base model (pre-trained)
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)

# Step 4: Freeze base model layers
base_model.trainable = False

# Step 5: Add custom layers on top
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

# Step 6: Create final model
model = Model(inputs=base_model.input, outputs=output)

# Step 7: Compile the model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Step 8: Show model summary
model.summary()