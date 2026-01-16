import sys
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from utils.config import IMG_SIZE, MODEL_PATH

model = load_model(MODEL_PATH)

img_path = sys.argv[1]

img = image.load_img(img_path, target_size=IMG_SIZE)
img = image.img_to_array(img) / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)[0][0]

print("Dog 🐶" if pred > 0.5 else "Cat 🐱")
