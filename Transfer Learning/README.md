# Dogs vs. Cats Transfer Learning with VGG16 (Keras)

Practice transfer learning notebooks on the `salader/dogsvscats` dataset (`20,000` training images and `5,000` test images across `2` classes; `(150, 150, 3)` input, `batch_size=32`, `10` epochs, `binary_crossentropy`) comparing partial fine-tuning of `block5` against a frozen `VGG16` base with data augmentation[cite: 9, 10].

## Notebooks & Logged Metrics
- **Block-5 Fine-Tuning (`vgg16_block5_finetuning.ipynb`):**[cite: 9]
  - **Setup:** ImageNet `VGG16` (`include_top=False`) with `block5_conv1` through `block5_pool` trainable (`7,079,424` trainable / `7,635,264` non-trainable / `14,714,688` total params), `Flatten` → `Dense(256, relu)` → `Dense(1, sigmoid)`, `RMSprop(learning_rate=1e-5)`[cite: 9]
  - **Epoch 1 (`625` steps):** `accuracy: 0.8992`, `loss: 0.2347`, `val_accuracy: 0.9364`, `val_loss: 0.1543`[cite: 9]
  - **Epoch 10 (`625` steps):** `accuracy: 0.9986`, `loss: 0.0058`, `val_accuracy: 0.9524`, `val_loss: 0.1546`[cite: 9]

- **Frozen Base + Augmentation (`vgg16_frozen_augmentation.ipynb`):**[cite: 10]
  - **Setup:** ImageNet `VGG16` (`include_top=False`, `14,714,688` params) with `conv_base.trainable = False`, `ImageDataGenerator` (`rescale=1./255`, `shear_range=0.2`, `zoom_range=0.2`, `horizontal_flip=True`), `Flatten` → `Dense(256, relu)` → `Dense(1, sigmoid)`, `optimizer="adam"`[cite: 10]
  - **Epoch 1 (`625` steps):** `accuracy: 0.8698`, `loss: 0.3061`, `val_accuracy: 0.9072`, `val_loss: 0.2171`[cite: 10]
  - **Epoch 10 (`625` steps):** `accuracy: 0.9360`, `loss: 0.1537`, `val_accuracy: 0.9174`, `val_loss: 0.2032`[cite: 10]

## How to Run
1. Install dependencies: `pip install -r requirements.txt`[cite: 9, 10]
2. Run either `vgg16_block5_finetuning.ipynb` or `vgg16_frozen_augmentation.ipynb` in Google Colab with GPU runtime and Kaggle credentials configured (datasets and weights are excluded from Git; models train dynamically with no pre-saved weight files)[cite: 9, 10].
