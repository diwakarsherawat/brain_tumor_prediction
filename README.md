# 🧠 Brain Tumor Classification using CNN

A deep learning project that detects and classifies brain tumors from MRI images into one of four categories using a Convolutional Neural Network (CNN). This project includes a Streamlit web app for real-time predictions.

---

## 📘 Overview

This project uses MRI images to classify brain tumors into:

- **Glioma**
- **Meningioma**
- **Pituitary**
- **No Tumor**

The goal is to aid in the early detection and categorization of brain tumors using deep learning models trained on real MRI scans.

---

## 🧠 Dataset

The dataset used is from [Kaggle - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset), which contains MRI images classified into four categories:

- `glioma_tumor/`
- `meningioma_tumor/`
- `pituitary_tumor/`
- `no_tumor/`


## 🧬 Model Architecture

The model is built using **TensorFlow/Keras** with a Convolutional Neural Network (CNN). Below is a simplified architecture:

- Input: 150x150 RGB image
- Conv2D → MaxPooling2D
- Conv2D → MaxPooling2D
- Flatten
- Dense (Fully Connected)
- Dropout
- Output: Softmax (4 classes)

Model saved as: `brain_tumor_cnn_model.h5`

---

## ⚙️ Tech Stack

- 🧠 **TensorFlow / Keras** — Model building and training
- 📊 **NumPy / PIL** — Image processing
- 🌐 **Streamlit** — Frontend UI
- ☁️ **Google Drive** — Model hosting
- 🔗 **Google Colab** — Training/Testing workspace

---

## 🚀 How It Works

1. User uploads an MRI image via the Streamlit web app.
2. Image is resized to 150x150 and normalized.
3. CNN model makes prediction.
4. Output is displayed showing predicted tumor type.

---

## 🌐 Web App

🚀 [Click here to try the app](https://braintumorprediction-74cqsqfdxr3mec8hnwcnps.streamlit.app/)


## 🔮 Future Improvements

- Improve model accuracy using data augmentation or advanced CNN architectures.
- Include batch image classification.
- Enable mobile support for the web app.



