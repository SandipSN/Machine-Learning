import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv('penguins_imputed.csv', index_col=[0])

def one_hot(df):
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    encode_cols = ['sex', 'island']
    df_cols = pd.DataFrame(OH_encoder.fit_transform(df[encode_cols]))
    df_cols.index = df.index
    df2 = df.drop(encode_cols, axis=1)
    df2 = pd.concat([df2, df_cols], axis=1)
    df2 = df2.reset_index(drop=True)
    df2 = df2.dropna(axis='columns')
    return df2

# Separate target from predictors
def split(data):
    y = data.species
    X = data.drop(['species'], axis=1)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
    return X_train, X_valid, y_train, y_valid

#scoring
def classification_score(X_train, X_valid, y_train, y_valid):
    model = RandomForestClassifier(random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return accuracy_score(y_valid, preds)

data_OH = one_hot(data)

target_mapper = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2}
def target_encode(val):
    return target_mapper[val]

data_OH['species'] = data_OH['species'].apply(target_encode)

X_train_1, X_valid_1, y_train, y_valid = split(data_OH)

#print(X_train_1.head())

#print(classification_score(X_train_1, X_valid_1, y_train, y_valid))

model = RandomForestClassifier(random_state=0)
model.fit(X_train_1, y_train)
#test = np.array([50.6,19.4,193.0,3800.0, 0, 1, 0, 1, 0]).reshape(1, -1)
#print(model.predict(test))

# Saving the model
import pickle
pickle.dump(model, open('penguins_clf.pkl', 'wb'))