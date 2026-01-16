from sklearn.utils.class_weight import compute_class_weight
from utils.config import *
from utils.data_loader import get_data_generators
from utils.model_builder import build_model

train_data, val_data, _ = get_data_generators(
    TRAIN_DIR, VAL_DIR, TEST_DIR
)

model = build_model()
model.summary()
classes = np.unique(train_data.classes)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=train_data.classes
)

class_weights = dict(zip(classes, class_weights))
print("Class Weights:", class_weights)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weights
)

model.save(MODEL_PATH)
print("✅ Model saved:", MODEL_PATH)
