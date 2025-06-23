# 🩺 Pneumonia Detection from Chest X-Rays using Deep Learning

This project uses **deep learning with transfer learning (ResNet50)** to classify chest X-ray images as either **PNEUMONIA** or **NORMAL**. It leverages preprocessing, augmentation, fine-tuning, and class balancing to achieve high performance.

---

## 📦 Dataset

The dataset is the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle. It contains:

- `NORMAL` class: healthy chest X-rays
- `PNEUMONIA` class: chest X-rays showing pneumonia (bacterial or viral)

**Classes are imbalanced** (more PNEUMONIA cases than NORMAL).

### 📥 How I downloaded it:

```bash
# Step 1: Authenticate
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia

# Step 2: Unzip
unzip chest-xray-pneumonia.zip
🧠 Model Architecture
This project uses ResNet50 from tensorflow.keras.applications with the following customizations:

✅ Input shape: (224, 224, 3)

✅ Converted grayscale to 3-channel RGB images

✅ Used preprocess_input from ResNet50 for consistent preprocessing

✅ Fine-tuned top layers (unfreeze_layers = 40)

✅ Added dropout and L2 regularization

✅ Used GlobalAveragePooling2D, Dense, and sigmoid for binary output

🧪 Preprocessing
Resized all images to 224x224

Converted grayscale images to RGB (3-channel) before feeding to ResNet

Applied preprocess_input (ImageNet standardization)

Used ImageDataGenerator for real-time data augmentation:

Rotation

Width/height shift

Zoom

Horizontal flip📊 Training Details
✅ Loss Function: binary_crossentropy

✅ Optimizer: Adam with learning rate tuning

✅ Metrics: Accuracy, Precision, Recall

✅ Class imbalance handled using class_weight

📈 Results
Metric	Training	Validation
Accuracy	~98.7%	~90.8%
Precision	~99.7%	~89.9%
Recall	~98.4%	~96.2%
Final Loss	~0.06	~0.31

📷 Predictions
Evaluated the model on a sample of 20 NORMAL and 20 PNEUMONIA images. Results show:

✅ Nearly perfect classification

✅ High-confidence predictions (close to 0.0 or 1.0)

✅ Visual inspection showed correct classifications in all tested cases

<!-- Replace with your image path -->

final_model.save("pneumonia_detection_model.keras")

🚀 Future Improvements
Add Grad-CAM for visual explanation

Test on cross-hospital datasets

Deploy using FastAPI or Streamlit

Balance recall vs precision for clinical sensitivity
