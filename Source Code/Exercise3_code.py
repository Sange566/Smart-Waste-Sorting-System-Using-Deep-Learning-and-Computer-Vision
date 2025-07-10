import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 📁 Dataset
dataset_path = r"C:\Users\Sange\OneDrive\Documents\Python\Garbage classification\Garbage classification"
img_height, img_width = 160, 160
batch_size = 32
epochs = 20

# 📈 Data Augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=40,
    zoom_range=0.3,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    brightness_range=[0.6, 1.4],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False  # Important for consistent prediction ordering
)

class_names = list(train_data.class_indices.keys())
num_classes = len(class_names)

# 📦 Transfer Learning Model
base_model = MobileNetV2(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')
base_model.trainable = True

# 🧠 Custom Classifier Head
inputs = Input(shape=(img_height, img_width, 3))
x = base_model(inputs, training=True)
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 📉 Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1),
    ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True, verbose=1)
]

# 🚀 Training
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    callbacks=callbacks
)

# 🔁 Fine-tuning
# Unfreeze all layers in the base model for fine-tuning
base_model.trainable = True

# Optional: Freeze batch normalization layers for stability during fine-tuning
for layer in base_model.layers:
    if isinstance(layer, BatchNormalization):
        layer.trainable = False

# Re-compile the model with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("\n🔁 Fine-tuning the model...\n")
history_finetune = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,  # You can adjust this as needed
    callbacks=callbacks
)


# 🧪 Evaluation
val_loss, val_acc = model.evaluate(val_data)
print(f"✅ Final Validation Accuracy: {val_acc * 100:.2f}%")

# 💬 Classification Report and Confusion Matrix
val_data.reset()
y_true = val_data.classes
y_pred_proba = model.predict(val_data, verbose=1)
y_pred = np.argmax(y_pred_proba, axis=1)

# Classification report
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# 💾 Save model
model.save("garbage_classifier_95_target.h5")
print("Model saved at 'garbage_classifier_95_target.h5'")
