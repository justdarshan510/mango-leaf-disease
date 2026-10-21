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
    data_root = EXTRACT_DIR
    subdirs = [os.path.join(data_root, d) for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    
    # Often ZIP files contain a single root folder wrapping the classes
    if len(subdirs) == 1:
        data_root = subdirs[0]
        
    print(f"Loading dataset from: {data_root}")
    
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 10 # Set to 10 for a relatively quick local training loop
    
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
    
    print("Building MobileNetV2 from SCRATCH (weights=None)...")
    base_model = applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None # Train entirely from scratch
    )
    
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(len(class_names), activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
                  
    print("Starting training...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        ]
    )
    
    print(f"Saving model to {MODEL_SAVE_PATH}")
    model.save(MODEL_SAVE_PATH)
    print("Training complete and model saved.")

if __name__ == "__main__":
    extract_dataset()
    train_model()
