import zipfile
import json
import os

model_path = "e:/mango leaf disese/mango_correct_classes_model.keras"
try:
    with zipfile.ZipFile(model_path, 'r') as z:
        if 'config.json' in z.namelist():
            with z.open('config.json') as f:
                config = json.load(f)
                
                # Check for Functional vs Sequential
                layers = []
                if config.get("class_name") == "Sequential":
                    layers = config["config"]["layers"]
                elif config.get("class_name") == "Functional":
                    layers = config["config"]["layers"]
                else:
                    print("Unknown model type:", config.get("class_name"))
                    
                print(f"Total layers: {len(layers)}\n")
                
                print("--- First 5 Layers ---")
                for i in range(min(5, len(layers))):
                    print(json.dumps(layers[i], indent=2))
                    print("\n")
                    
except Exception as e:
    print(f"Error: {e}")
