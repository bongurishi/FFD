import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import pandas as pd

# Paths
MODEL_PATH = "AlexNet_final.keras"
TEST_DATASET_DIR = "evaluation_dataset"   # change path if needed
IMG_HEIGHT, IMG_WIDTH = 227, 227  # CHANGED from 128 to 227
BATCH_SIZE = 32
CLASS_NAMES = ['Fresh', 'Rotten']

# Load model
print(" Loading model...")
try:
    model = load_model(MODEL_PATH)
    print(" Model loaded successfully!")
except Exception as e:
    print(f" Error loading model: {e}")
    exit()

# Load test data
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    TEST_DATASET_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),  # Now matches training size
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print(f" Found {test_generator.samples} images in test dataset")
print(f" Class indices: {test_generator.class_indices}")

# Evaluate model
print(" Evaluating on test data...")
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(f" Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f" Test Loss: {test_loss:.4f}")

# Predictions
print(" Making predictions...")
y_pred_probs = model.predict(test_generator, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f" Prediction Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Confusion matrix
print(" Generating confusion matrix...")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=CLASS_NAMES, 
            yticklabels=CLASS_NAMES,
            cbar_kws={'label': 'Number of Images'})
plt.xlabel("Predicted Label", fontsize=12, fontweight='bold')
plt.ylabel("True Label", fontsize=12, fontweight='bold')
plt.title("Confusion Matrix - AlexNet Model (227x227)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix_alexnet.png', dpi=300, bbox_inches='tight')
plt.show()

# Detailed classification report
print("\n Detailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# Additional metrics
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print(f"\n Additional Metrics:")
print(f" Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f" Recall: {recall:.4f} ({recall*100:.2f}%)")
print(f" F1-Score: {f1:.4f} ({f1*100:.2f}%)")

# Class-wise accuracy
class_correct = np.zeros(len(CLASS_NAMES))
class_total = np.zeros(len(CLASS_NAMES))

for i in range(len(CLASS_NAMES)):
    idx = (y_true == i)
    class_correct[i] = np.sum(y_pred[idx] == y_true[idx])
    class_total[i] = np.sum(idx)

print(f"\n Class-wise Accuracy:")
for i, class_name in enumerate(CLASS_NAMES):
    if class_total[i] > 0:
        class_accuracy = class_correct[i] / class_total[i]
        print(f"   {class_name}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%) - {int(class_correct[i])}/{int(class_total[i])} correct")
    else:
        print(f"   {class_name}: No samples found")

print(f"\n Evaluation completed successfully!")
print(f" Overall Test Accuracy: {test_accuracy*100:.2f}%")
