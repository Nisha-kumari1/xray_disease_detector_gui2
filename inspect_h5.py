import os
import tensorflow as tf

model_path = os.path.abspath("models/xray_final_model.h5")
print(f"Loading model from: {model_path}")

try:
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # We don't want to print the whole summary if it's huge, just the types of layers or the name
    print(f"Model Name: {model.name}")
    print(f"Number of layers: {len(model.layers)}")
    
    # Check if any layer is a known base model
    transfer_bases = ['vgg', 'resnet', 'densenet', 'efficientnet', 'mobilenet', 'inception']
    
    found_base = False
    for layer in model.layers:
        if isinstance(layer, tf.keras.models.Model):
            print(f"Found nested model (Likely Transfer Learning Base): {layer.name}")
            found_base = True
        for base in transfer_bases:
            if base in layer.name.lower():
                print(f"Found layer matching '{base}': {layer.name}")
                found_base = True
                break
                
    if not found_base:
        print("No nested transfer learning base model detected. Might be a custom CNN.")
        
    print("\nFirst 10 layers:")
    for l in model.layers[:10]:
        print(f" - {l.name} ({l.__class__.__name__})")
        
    print("\nLast 5 layers:")
    for l in model.layers[-5:]:
        print(f" - {l.name} ({l.__class__.__name__})")
        
except Exception as e:
    print(f"Error loading model: {e}")
