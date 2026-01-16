import numpy as np
from tensorflow.keras.models import load_model
from utils.config import *
from utils.data_loader import get_data_generators
from utils.metrics import print_metrics

_, _, test_data = get_data_generators(
    TRAIN_DIR, VAL_DIR, TEST_DIR
)

model = load_model(MODEL_PATH)

preds = model.predict(test_data)
y_pred = (preds > 0.5).astype(int)
y_true = test_data.classes

print_metrics(y_true, y_pred)
