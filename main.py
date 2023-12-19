import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow import image
import numpy as np

# Load the saved model
model = load_model('accident_detection_model.h5')

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(250, 250))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def classify_image(image_path):
    try:
        processed_image = preprocess_image(image_path)
        predictions = model.predict(processed_image)
        if predictions[0][0] > 0.5: 
            return "Accident"
        else:
            return "Non-Accident"
    except Exception as e:
        print("Error classifying the image:", e)
        return "Error"

image_path = 'data/train/Non Accident/5_11.jpg'

result = classify_image(image_path)
print("The image is classified as:", result)
