import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Force CPU optimization
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

TRAIN_DIR = "cashew_cropped_dataset/train"
VAL_DIR = "cashew_cropped_dataset/valid"
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 5

def main():
    if not os.path.exists(TRAIN_DIR) or len(os.listdir(TRAIN_DIR)) == 0:
        print(f"Error: Crop directory {TRAIN_DIR} not found. Run 'create_cashew_crop_dataset.py' first.")
        return

    print("Loading Cashew Cropped Dataset...")
    train_ds = image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42
    )

    val_ds = image_dataset_from_directory(
        VAL_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    class_names = train_ds.class_names
    print(f"Detected Cashew Classes: {class_names}")

    # Preprocessing
    def preprocess(images, labels):
        return tf.keras.applications.mobilenet_v2.preprocess_input(images), labels

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(preprocess).cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.map(preprocess).cache().prefetch(buffer_size=AUTOTUNE)

    print("Building MobileNetV2 Architecture for Cashew Classification...")
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
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

    print("\nStarting Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # Save Keras model
    os.makedirs("cashew_models", exist_ok=True)
    keras_path = "cashew_models/cashew_mobilenet_model.keras"
    model.save(keras_path)
    print(f"\n✅ Model saved to: {keras_path}")

    # Save classes
    with open("cashew_models/cashew_classes.txt", "w", encoding="utf-8") as f:
        f.write(str(class_names))
    print("✅ Classes saved to cashew_models/cashew_classes.txt")

    # Export TFLite model
    print("\nConverting to Google LiteRT (TFLite) for Edge Deployment...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_path = "cashew_models/cashew_model.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    tflite_size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
    print(f"✅ LiteRT edge model exported to: {tflite_path} ({tflite_size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
