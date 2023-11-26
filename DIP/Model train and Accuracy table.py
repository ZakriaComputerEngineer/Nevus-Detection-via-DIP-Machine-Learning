import os
import numpy as np
from PIL import Image
import pandas as pd
from tabulate import tabulate
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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
        hog_features = hog(
            gray_image, pixels_per_cell=(16, 16), cells_per_block=(2, 2), orientations=9
        )

        # Store the computed features and corresponding label
        features.append(hog_features)
        labels.append(label)

# Split the data into 80% for training and 20% for testing
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# Scale the data
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
test_features_scaled = scaler.transform(test_features)

classifiers = [
    DecisionTreeClassifier(),
    LogisticRegression(max_iter=1000),
    AdaBoostClassifier(),
    KNeighborsClassifier(),
    RandomForestClassifier(n_estimators=100, random_state=42),
    GridSearchCV(SVC(), {'C': [0.1, 1, 10, 100], 'kernel': [
                 'linear', 'rbf', 'poly'], 'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]}, cv=5)
]

results = []

# Define class names
class_names = {
    0: "Atypical nevus",
    1: "Common nevus",
    2: "Melanoma"
}

for classifier in classifiers:
    # Train the classifier using the training data
    if isinstance(classifier, GridSearchCV):
        classifier.fit(train_features_scaled, train_labels)
    else:
        classifier.fit(train_features, train_labels)

    # Use the trained model to make predictions on the test data
    if isinstance(classifier, GridSearchCV):
        test_predictions = classifier.best_estimator_.predict(
            test_features_scaled)
    else:
        test_predictions = classifier.predict(test_features)

    # Calculate overall accuracy of the solution
    accuracy = accuracy_score(test_labels, test_predictions)
    print(f"Classifier: {classifier.__class__.__name__}")
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")

    # Generate a classification report with zero_division set to 1
    report = classification_report(
        test_labels, test_predictions, target_names=class_names.values(), zero_division=1
    )

    print(
        f"Classification Report - {classifier.__class__.__name__}:\n", report)
    print("\n")

    # Append results to the list
    results.append({
        "Classifier": classifier.__class__.__name__,
        "Overall Accuracy (%)": f"{accuracy * 100:.2f}"
    })

# Create a pandas DataFrame from the list of results
results_df = pd.DataFrame(results)

# Print the results as a table using tabulate
results_table = tabulate(results_df, headers='keys',
                         tablefmt='pretty', showindex=False)

# Print the table
print("\nResults:")
print(results_table)
