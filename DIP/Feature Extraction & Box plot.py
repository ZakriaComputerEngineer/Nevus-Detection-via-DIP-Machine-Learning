import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

Source_folder = r"C:\Users\786 COMPUTERS\Documents\Data_set"

# Define the paths to your dataset folders for the three classes
melanoma_folder = os.path.join(Source_folder, f'Melanoma')
common_nevus_folder = os.path.join(Source_folder, f'Common Nevus')
atypical_nevus_folder = os.path.join(Source_folder, f'Atypical Nevus')

# Initialize lists to store features and labels
features = []
labels = []

# Define labels for the three classes
class_labels = {
    melanoma_folder: 2,
    common_nevus_folder: 1,
    atypical_nevus_folder: 0
}

# Loop through the class folders and load images
for class_folder, label in class_labels.items():
    for image_filename in os.listdir(class_folder):
        image_path = os.path.join(class_folder, image_filename)
        image = Image.open(image_path)
        image = image.resize((224, 224))  # Resize to a common size if needed
        gray_image = rgb2gray(np.array(image))

        # Compute HOG features
        hog_features = hog(gray_image, pixels_per_cell=(
            8, 8), cells_per_block=(2, 2))

        # Store the computed features and corresponding label
        features.append(hog_features)
        labels.append(label)

# Split the data into 80% for training and 20% for testing
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.2, random_state=42)

# Visualizing the distribution of a specific feature against each class
feature_index = 0  # Specify the index of the feature you want to analyze

# Create a dictionary to map class labels to class names
class_names = {0: "Atypical nevus", 1: "Common nevus", 2: "Melanoma"}

# Separate the training data and labels by class
class_data = {label: [] for label in set(train_labels)}
for i, label in enumerate(train_labels):
    class_data[label].append(train_features[i][feature_index])

# Create box plots to visualize feature distribution by class
plt.figure(figsize=(10, 6))
plt.boxplot(class_data.values(), labels=[
            class_names[label] for label in class_data.keys()])
plt.xlabel('Classes')
plt.ylabel('Feature Value')
plt.title('Box Plot for Feature {}'.format(feature_index))
plt.show()

# Create scatter plots to visualize feature distribution by class
plt.figure(figsize=(10, 6))
for label, data in class_data.items():
    plt.scatter([class_names[label]] * len(data),
                data, label=class_names[label])
plt.xlabel('Classes')
plt.ylabel('Feature Value')
plt.title('Scatter Plot for Feature {}'.format(feature_index))
plt.legend()
plt.show()

# Analysis and Thresholding:
# You can visually inspect the plots to find optimal threshold values or conditions that best segregate the cases.
# Modify the code as needed to apply your specific thresholding or conditions.
