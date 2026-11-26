# Dogs vs. Cats Image Classification (Keras CNN)

Practice binary image classification project using a TensorFlow/Keras Convolutional Neural Network (`Conv2D`, `BatchNormalization`, `MaxPool2D`, and `Dropout`) to classify cat and dog images[cite: 3].

## Dataset & Model Specs
- **Dataset:** `salader/dogsvscats` (`20,000` training files and `5,000` test files across `2` classes; excluded from repository)[cite: 3]
- **Input Pipeline:** Resized to `(256, 256)`, `batch_size=32` (`625` training steps per epoch), normalized via `image / 255.0` as `tf.float32`[cite: 3]
- **Architecture:** `3` conv blocks (`32`, `64`, `128` `3x3` `Conv2D` + `BatchNormalization` + `2x2` `MaxPool2D`) → `Flatten` (`115,200`) → `Dense(128, relu)` + `Dropout(0.1)` → `Dense(64, relu)` + `Dropout(0.1)` → `Dense(1, sigmoid)`[cite: 3]
- **Parameters:** `14,848,193` total (`56.64 MB`), `14,847,745` trainable, `448` non-trainable (`1.75 KB`)[cite: 3]
- **Training Metrics (`10` epochs, `Adam`, `binary_crossentropy`):**
  - **Epoch 1:** `accuracy: 0.5821`, `loss: 1.3185`, `val_accuracy: 0.5956`, `val_loss: 0.6658`[cite: 3]
  - **Epoch 10:** `accuracy: 0.9714`, `loss: 0.0808`, `val_accuracy: 0.8250`, `val_loss: 0.7394`[cite: 3]
- **Sample Inference:** Tested on `/content/cat_2.jpg` (`(550, 550, 3)` resized to `(256, 256, 3)`)[cite: 3]

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run `dogs_vs_cats_cnn.ipynb` in Google Colab with GPU runtime enabled and Kaggle credentials configured (trains dynamically with no pre-saved model file)[cite: 3].