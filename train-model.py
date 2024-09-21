import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define paths
train_dir = 'dataset/train'
validation_dir = 'dataset/validation'
test_dir = 'dataset/test'

# Image parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    shear_range=0.2
)

validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators with error checking
def create_generator(directory, datagen, subset=None):
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    classes = os.listdir(directory)
    if len(classes) != 2:
        raise ValueError(f"Expected 2 class subdirectories (asbestos, non_asbestos), found {len(classes)}")
    
    print(f"Classes found in {directory}: {classes}")
    
    generator = datagen.flow_from_directory(
        directory,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True if subset == 'training' else False,
        subset=subset
    )
    
    print(f"Generator for {directory} created successfully.")
    print(f"Number of samples: {generator.samples}")
    print(f"Class indices: {generator.class_indices}")
    
    if generator.samples == 0:
        raise ValueError(f"No images found in {directory}")
    
    return generator

try:
    train_generator = create_generator(train_dir, train_datagen, subset='training')
    validation_generator = create_generator(validation_dir, validation_datagen)
    test_generator = create_generator(test_dir, test_datagen)
except Exception as e:
    print(f"Error creating data generators: {str(e)}")
    raise

# Check if we have enough data to train
if train_generator.samples < BATCH_SIZE:
    raise ValueError(f"Not enough training samples. Found {train_generator.samples}, need at least {BATCH_SIZE}")

if validation_generator.samples == 0:
    print("Warning: No validation data found. Training will proceed without validation.")

# Build the Model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile the Model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Train the Model
epochs = 10
try:
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=epochs,
        validation_data=validation_generator if validation_generator.samples > 0 else None,
        validation_steps=validation_generator.samples // BATCH_SIZE if validation_generator.samples > 0 else None,
        verbose=1
    )
except Exception as e:
    print(f"Error during model training: {str(e)}")
    raise

# Evaluate the Model
if test_generator.samples > 0:
    test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // BATCH_SIZE)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
else:
    print("Warning: No test data found. Skipping evaluation.")

# Save the Model
model.save('models/asbestos_detector.h5')
print("Model saved to 'models/asbestos_detector.h5'")

# Function to make predictions
def predict_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    img_array /= 255.  # Rescale the image

    prediction = model.predict(img_array)[0][0]
    print(f"Probability of asbestos: {prediction*100:.2f}%")
    print(f"Classification: {'Asbestos' if prediction > 0.5 else 'Non-Asbestos'}")

# Example usage:
# predict_image('path/to/your/image.jpg')