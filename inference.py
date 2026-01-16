import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from utils.config import *

# ---- Argument check ----
if len(sys.argv) < 2:
    print("❌ Usage: python inference.py <image_path>")
    sys.exit(1)

img_path = sys.argv[1]

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load and preprocess image
img = image.load_img(img_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Prediction
prob = model.predict(img_array)[0][0]

label = "Dog" if prob > 0.5 else "Cat"

print(f"🐾 Prediction: {label}")
print(f"🔢 Probability: {prob:.4f}")
print("Dog 🐶" if prob > 0.5 else "Cat 🐱")
