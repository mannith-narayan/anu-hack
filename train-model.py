import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define paths and parameters
base_dir = 'dataset'
asbestos_dir = os.path.join(base_dir, 'Asbestos')
not_asbestos_dir = os.path.join(base_dir, 'Not Asbestos')

IMAGE_SIZE = (224, 224)  # MobileNetV2 default input size
BATCH_SIZE = 8
EPOCHS = 30

# Function to get all image paths and labels
def get_data(directory):
    data = []
    labels = []
    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                data.append(img_path)
                labels.append(class_name)
    return np.array(data), np.array(labels)

# Get all image paths and labels
all_data, all_labels = get_data(base_dir)

# Split the data into train, validation, and test sets (60:20:20)
train_data, temp_data, train_labels, temp_labels = train_test_split(all_data, all_labels, test_size=0.4, random_state=42)
val_data, test_data, val_labels, test_labels = train_test_split(temp_data, temp_labels, test_size=0.5, random_state=42)

# Data Augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

# Only rescaling for validation and test data
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
def create_generator(data, labels, datagen):
    df = pd.DataFrame({'filename': data, 'class': labels})
    return datagen.flow_from_dataframe(
        dataframe=df,
        x_col='filename',
        y_col='class',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

train_generator = create_generator(train_data, train_labels, train_datagen)
validation_generator = create_generator(val_data, val_labels, valid_datagen)
test_generator = create_generator(test_data, test_labels, test_datagen)

print(f"Found {len(train_data)} training images")
print(f"Found {len(val_data)} validation images")
print(f"Found {len(test_data)} test images")

# Create the base model from the pre-trained MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
base_model.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

# Combine the base model and custom layers
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_data) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=len(val_data) // BATCH_SIZE,
    callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
    verbose=1
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_data) // BATCH_SIZE)
print(f'Test Accuracy: {test_accuracy*100:.2f}%')



# Save the model
model.save('models/asbestos_detector.h5')
print("Model saved to 'models/asbestos_detector.h5'")

# Function to make predictions
def predict_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)
    img_array /= 255.

    prediction = model.predict(img_array)[0][0]
    print(f"Probability of asbestos: {prediction*100:.2f}%")
    print(f"Classification: {'Asbestos' if prediction > 0.5 else 'Not Asbestos'}")

# Example usage:
# predict_image('path/to/your/image.jpg')

# Optional: Plot training history
import matplotlib.pyplot as plt

def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend()
    plt.title('Model Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Model Loss')
    
    plt.tight_layout()
    plt.show()

# Uncomment the following line to plot the training history
# plot_training_history(history)
