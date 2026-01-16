import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from utils.config import *
from utils.data_loader import get_data_generators
from utils.model_builder import build_model

# Load data
train_data, val_data, _ = get_data_generators(
    TRAIN_DIR, VAL_DIR, TEST_DIR
)

# Build model
model = build_model()
model.summary()

# ---- Class Weights (START SIMPLE) ----
classes = np.unique(train_data.classes)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=train_data.classes
)

class_weights = dict(zip(classes, class_weights))

print("Class Indices:", train_data.class_indices)
print("Class Distribution:", np.bincount(train_data.classes))
print("Class Weights:", class_weights)

# Train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weights
)

# Save model
model.save(MODEL_PATH)
print("✅ Model saved:", MODEL_PATH)
