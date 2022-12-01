# Penguin_Classifier_App

**https://penguins-classifier-st.herokuapp.com/**

## Languages Used
  **Python**

## Description
A web app that enables a user to input data about various penguin features in order to determine it's species. The data is sourced from the palmerpenguins library by Allison Horst, available here: https://github.com/allisonhorst/palmerpenguins.

**Penguins_Data_Cleaning.ipynb** : This Jupyter notebook contains the inital exploratory data analysis, data cleaning and experimentation between dropping missing values or imputing them. It also contains the accuracy results of the classification model on each method. Finally, it exports the cleaned dataset to a csv (penguins_imputed.csv) used to train the final model.

**clf_model.py**: The final model used to classify the imputed dataset. This is then exported using the pickle package (to a file called penguins_clf.pkl) in order to be called directly from the app's code without having to rewrite the model there.

**clf_test.ipynb**: Is a test of the classification model used in the final application, to ensure results are as expected.

**Penguins_Classifier.py**: Finally the code for the web app is stored here.


## Credit
This app was built using the streamlit package following the tutorial from the Data Proffessor YouTube channel, available here: https://www.youtube.com/watch?v=Eai1jaZrRDs

However, this repo deviates in places from the above. The tutorial uses the pandas package (pandas.get_dummies method) to encode categorical columns, however I used scikit learns's one hot encoder package (sklearn.preprocessing.OneHotEncoder) to achieve this instead as it was a more up to date method and more appropriate for machine learning models. 

The tutorial also just dropped missing values from the data, whereas I used an imputation method to fill these instead. This yeilded better accuracy than the tutorial as a result.

## Next Steps and known bugs
- The final prediciton on the app displays as an array, where it should be in a table. This may be a problem with the target_mapper in the model file and how it is linking to the final app file. Need to investigate and fix

