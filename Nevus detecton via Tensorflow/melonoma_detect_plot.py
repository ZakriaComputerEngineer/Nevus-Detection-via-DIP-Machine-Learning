import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the saved model trained to classify specific classes
model = tf.keras.models.load_model("melanoma_detection_model.keras")

# Define a list of class labels corresponding to your specific classes
class_labels = ["Atypical Nevus", "Common Nevus", "Melanoma"]


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_and_plot(image_path):
    predicted_class, confidence = predict_specific_class(image_path)
    img = mpimg.imread(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(
        f"Predicted Class: {predicted_class}\nConfidence: {confidence:.2f}")
    probabilities = model.predict(preprocess_image(image_path))[0]

    # Plot the probability distribution as a bar chart
    plt.figure()
    plt.bar(class_labels, probabilities, color='blue')
    plt.xlabel('Classes')
    plt.ylabel('Probability')
    plt.title('Probability Distribution')
    plt.show()


def predict_specific_class(image_path):
    preprocessed_image = preprocess_image(image_path)
    predictions = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions)
    confidence = predictions[0][predicted_class_index]
    predicted_class_label = class_labels[predicted_class_index]
    return predicted_class_label, confidence


# Provide the path to the image you want to predict and plot
image_path = r"F:\WORK\python practice\pictures for test\image0.jpg"
predict_and_plot(image_path)
