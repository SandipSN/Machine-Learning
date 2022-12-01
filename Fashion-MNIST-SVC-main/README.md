# Fashion-MNIST-SVC

## Languages Used
Python

## Description

The Fashion MNIST dataset aims to replace the more common MNIST classification problem (digit recogniser). Instead here, the data consists of 28x28 pixel greyscale images of different items of clothing.

This notebook imports the Fashion MNIST dataset then normalises and reshapes the dataset ready for modelling. The model chosen here was the Support Vector Classifier from sklean's package and was employed to categorise the images into 10 classes. This model was able to achieve the following scores:

F1 score: 0.89
Accuracy score: 0.89
Precision score: 0.89
Recall Score: 0.89



## Credit
Original dataset is sourced from: https://github.com/zalandoresearch/fashion-mnist

I obtained the the training and test data from: https://www.kaggle.com/zalando-research/fashionmnist


## Next Steps and known bugs

Further work wil be done to better understand why the accuracy metrics were all identical and to try different models to compare which performs better.
