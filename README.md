# Deepfake Image Detector

This project implements a **Deepfake Image Classification System** using **Convolutional Neural Networks (CNN)** and **transfer learning** with **EfficientNetB0**. It is trained on the "Real and Fake Face Detection" dataset from Kaggle and provides a user-friendly web interface using **Gradio** for real-time classification.

---

## Table of Contents

* [Overview](#overview)
* [Technologies Used](#technologies-used)
* [Dataset](#dataset)
* [Model Architecture](#model-architecture)
* [Training Details](#training-details)
* [Performance](#performance)
* [Gradio Interface](#gradio-interface)
* [How to Run](#how-to-run)
* [Project Structure](#project-structure)
* [Future Work](#future-work)
* [License](#license)

---

## Overview

Deepfakes have emerged as a major concern in the digital age, with increasing use in misinformation and identity fraud. This project aims to build a deep learning-based classifier to distinguish between real and AI-generated (fake) human faces using image classification.

---

## Technologies Used

* Python
* TensorFlow / Keras
* EfficientNetB0 (Transfer Learning)
* OpenCV
* Scikit-learn
* Matplotlib
* Gradio
* KaggleHub

---

## Dataset

* **Name:** Real and Fake Face Detection
* **Source:** [Kaggle Dataset - ciplab/real-and-fake-face-detection](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection)
* **Contents:**

  * `training_real`: Real human face images
  * `training_fake`: AI-generated human face images

The dataset was downloaded and prepared using the `kagglehub` library.

---

## Model Architecture

* **Backbone:** EfficientNetB0 (pre-trained on ImageNet)
* **Custom Layers:**

  * Global Average Pooling
  * Dense Layer with L2 Regularization
  * Batch Normalization
  * Dropout
  * Softmax Output Layer (2 classes: Real, Fake)

Only the top 20 layers of EfficientNetB0 were made trainable to fine-tune the model.

---

## Training Details

* **Input Size:** 224x224 RGB images
* **Augmentation:** ImageDataGenerator with rotation, shift, zoom, flip, brightness, and channel shift
* **Optimizer:** Adam
* **Loss Function:** Categorical Crossentropy with label smoothing
* **Callbacks Used:**

  * ModelCheckpoint
  * EarlyStopping
  * ReduceLROnPlateau
* **Class Weighting:** Used to handle class imbalance
* **Epochs:** 10
* **Batch Size:** 16

---

## Performance

Training and validation accuracy and loss are plotted using `matplotlib`. The model is saved as `deepfake_cnn.h5` for future inference.

---

## Gradio Interface

A lightweight and interactive Gradio UI allows users to upload an image and receive a prediction of whether the face is real or fake.

* **Input:** Image file (PNG, JPG)
* **Output:** Class probabilities for Real and Fake
* **Framework:** Gradio

---

## How to Run

1. **Install Dependencies:**

   ```bash
   pip install kagglehub gradio tensorflow matplotlib
   ```

2. **Download Dataset and Train the Model:**

   Use the provided script to download, preprocess, augment, and train the model.

3. **Launch Gradio App:**

   ```bash
   python your_script.py
   ```

---

## Project Structure

```
.
├── real/                       # Copied real face images
├── fake/                       # Copied fake face images
├── deepfake_cnn.h5             # Trained model
├── best_model.h5               # Best model checkpoint
├── README.md                   # Project documentation
```

---

## Future Work

* Add video input support for detecting deepfakes in videos
* Use larger or more diverse datasets for improved generalization
* Integrate explainable AI methods (e.g., Grad-CAM)
* Improve real-time inference speed using model quantization or TFLite

---

## License

## License

This project is licensed under the GNU General Public License v3.0.

You are free to use, modify, and distribute this software under the terms of the GPLv3.  
See the [LICENSE](LICENSE) file for full details.
