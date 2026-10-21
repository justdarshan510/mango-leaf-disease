import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Force CPU to avoid CUDA errors if GPU isn't properly configured
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Dataset configuration
DATASET_DIR = "dataset/Augmented Mango dataset"
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 5

print("Loading dataset...")
# Load dataset
train_dataset = image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_dataset = image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_dataset.class_names
print(f"Detected classes: {class_names}")

# Apply MobileNetV2 preprocessing mapping to the datasets
# This ensures that images are preprocessed EXACTLY the same way during training and inference.
def preprocess(images, labels):
    return tf.keras.applications.mobilenet_v2.preprocess_input(images), labels

train_dataset = train_dataset.map(preprocess)
val_dataset = val_dataset.map(preprocess)

# Optimize for performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

print("Building MobileNetV2 Transfer Learning Model...")
# Create the base model from the pre-trained model MobileNetV2
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the base model
base_model.trainable = False

# Add custom classification head
inputs = tf.keras.Input(shape=(224, 224, 3))
# Note: we do NOT apply preprocess_input as a layer here because we already mapped it onto the dataset
# This guarantees that at inference time, we manually apply the exact same function.
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
outputs = Dense(len(class_names), activation='softmax')(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("Starting training...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)

# Save the model
model_path = 'local_mango_model.keras'
model.save(model_path)
print(f"Training complete. Model saved to {model_path}")

# Write class names to a file for reference
with open('classes.txt', 'w') as f:
    f.write(str(class_names))
print("Class names saved to classes.txt")
