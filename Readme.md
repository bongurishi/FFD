
# **Food Freshness Detection (FFD) Using Deep Learning**

---

## ** Project Overview**

This project aims to **detect the freshness of fruits** (Fresh vs Rotten) using **deep learning** (AlexNet CNN) and **image preprocessing techniques**. The project integrates a **user-friendly Streamlit web app** for real-time predictions and includes a **cost calculator** and **feedback module**.

**Applications**:

* Smart inventory management
* Food quality monitoring
* Mobile/web-based freshness detection tools

---

## ** Datasets**

### **1. Training Dataset (`final_dataset/`)**

* Contains labeled images: **Fresh / Rotten**
* Images processed for consistency:

  * Resize to **227×227** (AlexNet input size)
  * Normalize pixel values (0–1)
  * Optional grayscale and noise reduction
* **Data Augmentation** applied to prevent overfitting:

  * Rotation (±25°)
  * Width/Height shift
  * Shear & zoom
  * Horizontal flipping

### **2. Evaluation Dataset (`evaluation_dataset/`)**

* Used for **testing only**
* Images resized to **128×128**
* Only normalization applied (no augmentation)
* Provides real-world assessment of model performance
* Currently contains 12 images (mostly Rotten), which leads to **imbalanced evaluation metrics**

---

## ** Model: AlexNet Architecture**

| Layer   | Description                                                        |
| ------- | ------------------------------------------------------------------ |
| Conv2D  | 96 filters, 11×11 kernel, stride 4 → ReLU → BatchNorm → MaxPool    |
| Conv2D  | 256 filters, 5×5 kernel, padding=same → ReLU → BatchNorm → MaxPool |
| Conv2D  | 384 filters, 3×3 kernel, padding=same → ReLU                       |
| Conv2D  | 384 filters, 3×3 kernel, padding=same → ReLU                       |
| Conv2D  | 256 filters, 3×3 kernel, padding=same → ReLU → MaxPool             |
| Flatten | Flatten feature maps                                               |
| Dense   | 4096 neurons → ReLU → Dropout 0.5                                  |
| Dense   | 4096 neurons → ReLU → Dropout 0.5                                  |
| Dense   | 2 neurons → Softmax (Fresh / Rotten)                               |

**Training Details:**

* Optimizer: **Adam (LR=0.0001)**
* Loss: **Categorical Crossentropy**
* Metrics: **Accuracy**
* Callbacks:

  * **EarlyStopping** → stop training if validation loss doesn’t improve
  * **ModelCheckpoint** → save best model as `AlexNet_best_model.keras`
* Epochs: 2 (demo; can increase)
* Batch Size: 32

---

## ** Training Pipeline**
```
final_dataset → DataGenerator (Augmentation) → AlexNet → Best Model Saved
```

* **Data Generator** feeds batches of images to the model
* Augmentation improves generalization
* Model checkpoints save the best performing weights

---

## ** Evaluation**

* Load trained model: `AlexNet_best_model.keras`
* Load evaluation dataset via ImageDataGenerator (rescale only)
* Predict classes and compute metrics:

  * **Accuracy**
  * **Confusion Matrix**
  * **Classification Report (Precision, Recall, F1-score)**

**Example Classification Report**:

```
             precision    recall  f1-score   support

       Fresh       0.00      0.00      0.00         0
      Rotten       1.00      0.92      0.96        12

    accuracy                           0.92        12
   macro avg       0.50      0.46      0.48        12
weighted avg       1.00      0.92      0.96        12
```

**Observation**:

* Evaluation dataset is **imbalanced** → Fresh metrics are 0
* Accuracy seems high (0.92) but is misleading
* Recommendation: Include Fresh images for balanced evaluation

---

## ** Streamlit Web App**

### **Features**

1. **Multi-language support**: English, Hindi, Telugu
2. **Image input**: Upload or capture via camera
3. **Prediction**: Model predicts Fresh/Rotten with confidence
4. **Freshness estimation**:

   * Fresh → 5 days
   * Rotten → 1 day
5. **Cost calculator**:

   * Fresh: normal cost
   * Rotten: 50% discounted cost
6. **Feedback module**: Users can submit textual feedback

### **Workflow**

```
User Upload/Capture → Preprocess Image → Model Prediction → Show Class + Confidence
→ Estimate Freshness Days → Cost Calculator → Feedback Submission
```

**Tabs in App**:

1. **Welcome** → Select language
2. **Detection** → Upload/Capture + Prediction
3. **Calculator** → Cost estimation based on freshness
4. **Feedback** → Submit feedback

**Key Functions**:

* `preprocess_image(image)` → Resize + normalize + expand dims
* `estimate_days(predicted_class)` → Returns freshness duration
* `load_alexnet_model()` → Load model **once** with caching

---

## ** Project Flow Diagram**

```
             ┌─────────────────────────┐
             │     final_dataset       │
             │  (Training Images)      │
             └───────────┬────────────┘
                         │ Preprocessing + Augmentation
                         ▼
             ┌─────────────────────────┐
             │     Data Generator      │
             └───────────┬────────────┘
                         │
                         ▼
             ┌─────────────────────────┐
             │       AlexNet Model     │
             │  (5 Conv + 3 FC layers)│
             └───────────┬────────────┘
                         │ Save Best Model
                         ▼
             ┌─────────────────────────┐
             │ AlexNet_best_model.keras│
             └───────────┬────────────┘
                         │
         ┌───────────────┴─────────────────┐
         ▼                                 ▼
┌───────────────────────┐          ┌─────────────────────┐
│ evaluation_dataset    │          │ Streamlit App       │
│ (Testing Images)      │          │ Upload / Camera     │
└───────────┬───────────┘          │ Preprocess Image    │
            │                        │ Predict Class       │
            ▼                        │ Estimate Freshness  │
┌───────────────────────┐          │ Calculate Cost      │
│ Load Model & Predict  │          │ Feedback Submission │
└───────────┬───────────┘          └─────────────────────┘
            │
            ▼
┌───────────────────────┐
│ Confusion Matrix       │
│ Classification Report  │
└───────────────────────┘
```

---

## ** Requirements**

* Python packages:

```
tensorflow
numpy
pillow
opencv-python
streamlit
```

* Files: `AlexNet_best_model.keras`
* Datasets: `final_dataset` (train/val), `evaluation_dataset` (test)

---

## ** Contributors**

1. [BONGU RISHI](https://www.linkedin.com/in/bongurishi07/)
2. [SEEMA YADAV](https://www.linkedin.com/in/seema-yadav-b1b3a01a0/)
3. [JASIKA](https://www.linkedin.com/in/jasika-b1b3a01a0/)
4. [RAJVEER](https://www.linkedin.com/in/rajveer--b1b3a01a0/)
5. [BHUMIKA](https://www.linkedin.com/in/bhumika-b1b3a01a0/)

---

## ** Future Scope**

* Real-time detection via camera feed
* Multi-fruit classification
* Mobile deployment using **TensorFlow Lite** or **ONNX**
* Integration with inventory management systems

---

