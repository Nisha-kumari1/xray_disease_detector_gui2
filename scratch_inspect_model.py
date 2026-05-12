import tensorflow as tf
import sys

try:
    model = tf.keras.models.load_model('e:/AIML_Project/xray_disease_detector_gui2/models/xray_final_model.h5', compile=False)
    model.summary()
    print("\n--- Last Conv Layer ---")
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            print(f"Name: {layer.name}, Shape: {layer.output_shape}")
            break
except Exception as e:
    print("Error:", e)
    sys.exit(1)
