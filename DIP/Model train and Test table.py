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
from prettytable import PrettyTable


def load_images_and_features(dataset_folder):
    all_features, all_labels = [], []
    class_labels = {
        "Melanoma": 2,
        "Common Nevus": 1,
        "Atypical Nevus": 0
    }

    for class_name, label in class_labels.items():
        current_class_folder = os.path.join(dataset_folder, class_name)
        for image_filename in os.listdir(current_class_folder):
            image_path = os.path.join(current_class_folder, image_filename)
            image = Image.open(image_path)
            # Resize to a common size if needed
            image = image.resize((224, 224))
            gray_image = rgb2gray(np.array(image))

            hog_features = hog(gray_image, pixels_per_cell=(
                16, 16), cells_per_block=(2, 2), orientations=9)

            all_features.append(hog_features)
            all_labels.append(label)

    return all_features, all_labels


def split_data(features, labels):
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    return train_features, test_features, train_labels, test_labels


def train_model(train_features, train_labels, classifier):
    if isinstance(classifier, GridSearchCV):
        classifier.fit(train_features, train_labels)
        model = classifier.best_estimator_
    else:
        model = classifier
        model.fit(train_features, train_labels)
    return model


def evaluate_model(model, test_features, test_labels):
    predictions = model.predict(test_features)
    overall_accuracy = accuracy_score(test_labels, predictions)
    return overall_accuracy, predictions


def visualize_accuracy_table(test_labels, predictions, class_names):
    # Create a PrettyTable for displaying accuracy values
    accuracy_table = PrettyTable()
    accuracy_table.field_names = [
        'Image', 'True Label', 'Predicted Label', 'Accuracy']

    for i, true_label in enumerate(test_labels):
        predicted_label = predictions[i]
        is_correct = true_label == predicted_label
        accuracy_table.add_row(
            [f"Image {i+1}", class_names[true_label], class_names[predicted_label], is_correct])

    # Calculate overall average accuracy
    overall_accuracy = accuracy_score(test_labels, predictions)
    accuracy_table.add_row(
        ['', '', 'Overall Accuracy', f"{overall_accuracy:.2%}"])

    print(accuracy_table)


def main():
    dataset_folder = r"C:\Users\786 COMPUTERS\Documents\Data_set"
    features, labels = load_images_and_features(dataset_folder)

    class_names = {0: "Atypical Nevus", 1: "Common Nevus", 2: "Melanoma"}

    train_features, test_features, train_labels, test_labels = split_data(
        features, labels)

    classifiers = [
        DecisionTreeClassifier(),
        LogisticRegression(max_iter=1000),
        AdaBoostClassifier(),
        KNeighborsClassifier(),
        RandomForestClassifier(n_estimators=100, random_state=42),
        GridSearchCV(SVC(), {'C': [0.1, 1, 10, 100], 'kernel': [
                     'linear', 'rbf', 'poly'], 'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]}, cv=5)
    ]

    for classifier in classifiers:
        # Train your model
        model = train_model(train_features, train_labels, classifier)

        # Use your trained model to evaluate on the remaining 20% data
        overall_accuracy, predictions = evaluate_model(
            model, test_features, test_labels)

        print(f"\nClassifier: {classifier.__class__.__name__}")
        print(f"Overall Accuracy on 20% Test Data: {overall_accuracy:.2%}")

        # Visualize accuracy table
        visualize_accuracy_table(test_labels, predictions, class_names)


if __name__ == "__main__":
    main()
