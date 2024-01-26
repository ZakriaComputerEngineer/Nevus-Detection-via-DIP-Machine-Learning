import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

model = tf.keras.models.load_model("melanoma_detection_model.keras")

class_labels = ["Atypical Nevus", "Common Nevus", "Melanoma"]


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_specific_class(image_path):
    preprocessed_image = preprocess_image(image_path)
    predictions = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions)
    confidence = predictions[0][predicted_class_index]
    predicted_class_label = class_labels[predicted_class_index]
    return predicted_class_label, confidence


# Provide the path to the image you want to predict
image_path = r"F:\WORK\python practice\pictures for test\image1.jpg"

predicted_class, confidence = predict_specific_class(image_path)

print("Predicted Class:", predicted_class)
print("Confidence:", confidence)
