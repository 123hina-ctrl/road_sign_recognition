import gradio as gr
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load your best trained model
model = load_model('best_CNN_model.h5')

# ?? Replace this with your prediction function
def predict_sign(image):
    # Preprocess the image as done for the training data
    img = image.resize((32, 32))  # Resize to 32x32
    img = np.array(img)
    # Convert to grayscale (as done in cell v6OWqK7PMm6a)
    img_gray = np.sum(img/3, axis=2, keepdims=True)
    # Normalize (as done in cell d6VgAqHaM7kd)
    img_gray_norm = (img_gray - 32) / 32
    img_gray_norm = img_gray_norm.reshape(1, 32, 32, 1)  # reshape for prediction


    result = model.predict(img_gray_norm)
    pre