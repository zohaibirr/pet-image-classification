from utils.config import *
from utils.data_loader import get_data_generators
from utils.model_builder import build_model

train_data, val_data, _ = get_data_generators(
    TRAIN_DIR, VAL_DIR, TEST_DIR
)

model = build_model()
model.summary()

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

model.save(MODEL_PATH)
print("✅ Model saved:", MODEL_PATH)
