import os
import zipfile
import tensorflow as tf
from tensorflow.keras import layers, models, applications

ZIP_PATH = r"C:\Users\justd\Downloads\Augmented Mango dataset.zip"
EXTRACT_DIR = r"dataset"
MODEL_SAVE_PATH = "local_mango_model.keras"

def extract_dataset():
    if not os.path.exists(EXTRACT_DIR):
        print(f"Extracting {ZIP_PATH} to {EXTRACT_DIR}...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
        print("Extraction complete.")
    else:
        print("Dataset already extracted.")

def train_model():
    data_root = "dataset_cropped"
    if not os.path.exists(data_root):
        print(f"Dataset not found at {data_root}. Please run preprocess_dataset.py first.")
        return
        
    print(f"Loading dataset from: {data_root}")
    
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS_FEATURE_EXTRACTION = 5
    EPOCHS_FINE_TUNING = 5
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_root,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_root,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    class_names = train_ds.class_names
    print(f"Found classes: {class_names}")
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    print("Building MobileNetV2 with TRANSFER LEARNING (weights='imagenet')...")
    base_model = applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet' # Use pre-trained weights!
    )
    
    # Freeze the base model for initial training
    base_model.trainable = False
    
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ])
    
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False) # Ensure batchnorm stays in inference mode
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(len(class_names), activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
                  
    print("Starting Feature Extraction Training...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FEATURE_EXTRACTION,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)]
    )
    
    print("Starting Fine-Tuning...")
    base_model.trainable = True
    # Freeze bottom layers, unfreeze the top layers
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    # Recompile with a lower learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
                  
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINE_TUNING,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)]
    )
    
    print(f"Saving model to {MODEL_SAVE_PATH}")
    model.save(MODEL_SAVE_PATH)
    print("Training complete and model saved.")

if __name__ == "__main__":
    # extract_dataset() # Not needed anymore, reading from dataset_cropped
    train_model()
