"""
evaluation.py
--------------
Evaluates trained CNN model on test dataset using
Precision, Recall, F1-score, ROC-AUC and visual plots.

This script is CPU/GPU agnostic and uses NO pretrained models.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve
)

from utils.config import *
from utils.data_loader import get_data_generators

# ----------------------------------------------------
# Load test data
# ----------------------------------------------------
_, _, test_data = get_data_generators(
    TRAIN_DIR, VAL_DIR, TEST_DIR
)

# ----------------------------------------------------
# Load trained model
# ----------------------------------------------------
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded from:", MODEL_PATH)

# ----------------------------------------------------
# OLD / SIMPLE APPROACH (KEPT AS COMMENT)
# ----------------------------------------------------
# test_loss, test_acc = model.evaluate(test_data)
# print("Test Accuracy:", test_acc)

# ----------------------------------------------------
# Predictions
# ----------------------------------------------------
y_true = test_data.classes
y_probs = model.predict(test_data).ravel()

# Default threshold = 0.5 (can be tuned later)
y_pred = (y_probs >= 0.5).astype(int)

# ----------------------------------------------------
# Classification Metrics
# ----------------------------------------------------
print("\n📊 Classification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=list(test_data.class_indices.keys())
    )
)

# ----------------------------------------------------
# Confusion Matrix
# ----------------------------------------------------
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=list(test_data.class_indices.keys())
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# ----------------------------------------------------
# ROC Curve & AUC
# ----------------------------------------------------
fpr, tpr, _ = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

print(f"📈 ROC-AUC Score: {roc_auc:.4f}")

# ----------------------------------------------------
# Precision-Recall Curve
# ----------------------------------------------------
precision, recall, _ = precision_recall_curve(y_true, y_probs)

plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()

# ----------------------------------------------------
# Summary
# ----------------------------------------------------
print("✅ Evaluation completed successfully.")
