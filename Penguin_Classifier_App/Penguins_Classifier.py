import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

# ADD about button for below

st.write("""
# Penguin Species Prediction App
This app predicts the **Palmer Penguin** species!
Data obtained from the [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst.
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
        sex = st.sidebar.selectbox('Sex',('MALE','FEMALE'))
        culmen_length_mm = st.sidebar.slider('culmen length (mm)', 32.1,59.6,43.9)
        culmen_depth_mm = st.sidebar.slider('culmen depth (mm)', 13.1,21.5,17.2)
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0,231.0,201.0)
        body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
        data = {'island': island,
                'culmen_length_mm': culmen_length_mm,
                'culmen_depth_mm': culmen_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
penguins_raw = pd.read_csv('penguins_imputed.csv')
penguins = penguins_raw.drop(columns=['species'])
df = pd.concat([input_df, penguins],axis=0)

# Encoding of ordinal features in seperate columns

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

df = one_hot(df)
#df = df.dropna(axis='columns')
df = df[:1]


# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)


# Reads in saved classification model
load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

# Apply model to make predictions
#test = np.array([50.6,19.4,193.0,3800.0, 0, 1, 0, 1, 0]).reshape(1, -1)
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)
print(prediction)
st.subheader('Prediction')

#penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.write([prediction])


st.subheader('Prediction Probability')
st.write(prediction_proba)
