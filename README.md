# Nevus-Detection-via-DIP-Machine-Learning
Kindly read the problem statement first to understand my approaches, you need to download the ph2 folder from the link given below and give directory to the create_dataset code to generate format (class wise image seperation) for model genearation.

Dataset and Model file link: https://drive.google.com/drive/folders/1c8tH-UwQfXQkHwmLfSPPtWpcaVODaq17?usp=drive_link

DIP:

1) Data seperation and Classification
2) Feature extraction
3) Feature scalar value assigning for further graphical representation
4) Box plot graph
5) Analysis: Box plot shows variation among provided images in each of these 3 classes
6) using different models for detection and checking the accuracy:
  1) Decision Tree Classifier   47.50%
  2) Logistic Regression        55.00%
  3) AdaBoost Classifier        55.00%
  4) KNeighbors Classifier      55.00%
  5) Random Forest Classifier   60.00%
  6) Grid Search CV             55.00%
7) Per image accuracy table

ML:

1) Data set fornat for model training
2) threshold, feature extraction (hog) and epoche = 10 setting for model training (tensorflow)
3) preprocessing image
4) Dedection using trained model:
   1) Best Accuracy: 95.3%
   2) Worst Accuracy: 49.0%
