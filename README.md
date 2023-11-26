# Nevus-Detection-via-DIP-Machine-Learning
Kindly read the problem statement first to understand my approaches

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
