import tensorflow as tf
import numpy as np
from PIL import Image
import os

model = tf.keras.models.load_model("local_mango_model.keras")
CLASS_NAMES = ['Anthracnose', 'Die Black', 'Gall Midge', 'Healthy', 'Powdery Mildew']

def test_folder(folder_name):
    path = os.path.join("dataset", "Augmented Mango dataset", folder_name)
    files = os.listdir(path)[:5]
    print(f"--- Testing {folder_name} ---")
    for f in files:
        img_path = os.path.join(path, f)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        # Removed double preprocessing here!
        preds = model.predict(img_array, verbose=0)[0]
        idx = np.argmax(preds)
        print(f"  {f}: predicted {CLASS_NAMES[idx]} ({preds[idx]:.2f})")

test_folder("Gall Midge")
test_folder("Die Black")
test_folder("Anthracnose")
