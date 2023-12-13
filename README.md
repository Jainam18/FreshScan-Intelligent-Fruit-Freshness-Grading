# Freshness and Expiry Date Detection in Packaged Food Products


## Overview
This research project focuses on two innovative approaches for assessing fruit freshness and detecting the expiry date of packaged food products. The quality, taste, texture, and nutritional value of food are significantly influenced by freshness. The study combines computer vision techniques, supervised learning algorithms, and deep learning models to achieve accurate freshness grading of fruits and expiry date detection on various packaged products.

## Key Features
### Fruit Freshness Grading:

Utilizes YOLO V5, image processing, and supervised learning algorithms.
Classifies fruit images into Apples, Bananas, and Tomatoes.
Extracts 26 features from images for training freshness grading models.
Implements PCA for dimensionality reduction to enhance model performance.

### Expiry Date Detection:
Annotated datasets, PCA, image processing, and deep learning techniques are employed.
The deep learning model accurately predicts bounding boxes on images.
Optical Character Recognition (OCR) extracts expiry dates from cropped images.

## Results

### Freshness Grading:
For Apples, SVM achieves an impressive 94% accuracy in classifying fresh and not fresh fruits.
Random Forest achieves 68% accuracy in multi-grading categories of ripeness for Apples.
KNN achieves 93% accuracy for different degrees of ripeness for Bananas.
Random Forest Classifier achieves 89% accuracy in grading Tomatoes.

### Expiry Date Detection:
The system accurately predicts bounding boxes on images, enabling successful OCR for expiry date extraction.
