import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from PIL import Image

st.markdown(
    f"""
       <style>
       .stApp {{
           background-image: url("https://img.freepik.com/free-vector/blue-pink-halftone-background_53876-99004.jpg");
           background-attachment: fixed;
           background-size: cover;
           /* opacity: 0.3; */
       }}
       </style>
       """,
    unsafe_allow_html=True
)
st.markdown(
    "<h1 style='text-align: center; color:black ;font-size:45px ;'>PENGUINS PREDICTS APP !</h1>"
    "<i><p style='text-align: center; color:#4E4E50 ; font-size:18px ;'>Welcome to web app predicts the Palmer Penguin species</p>"
    , unsafe_allow_html=True
    
    )

img = Image.open("P2.png")
st.image(img)

st.sidebar.header('📁 User Input Features')

# train model
#uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
#st.sidebar.markdown("""
#[example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
#""")



def user_input_features():
    island = st.sidebar.selectbox('▶ ISLAND',('Biscoe','Dream','Torgersen'))
    sex = st.sidebar.selectbox('▶ SEX',('male','female'))
    bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1,59.6,43.9)
    bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1,21.5,17.2)
    flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0,231.0,201.0)
    body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
    data = {'island': island,
            'bill_length_mm': bill_length_mm,
            'bill_depth_mm': bill_depth_mm,
            'flipper_length_mm': flipper_length_mm,
            'body_mass_g': body_mass_g,
            'sex': sex}
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()


# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
penguins_raw = pd.read_csv('model-building/penguins_cleand.csv')
penguins = penguins_raw.drop(columns=['species'])
df = pd.concat([input_df,penguins],axis=0)

encode = ['sex','island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # the user input data

st.markdown(
    "<h1 style='text-align: #center; color:black ;font-size:27px ;'>📁 User Input features</h1>"
    , unsafe_allow_html=True   
    )


st.write('➥ Awaiting CSV file to be uploaded (currently using example input parameters)')
st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('▶ Prediction')
penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.write(penguins_species[prediction])

st.subheader('▶ Prediction Probability')
st.write(prediction_proba)


if st.button("Penguin data"):
    st.markdown(
        "<h1 style='text-align: #center; color:black ;font-size:30px ;'>➥ 📃 Describing the data </h1>"
        , unsafe_allow_html=True   
        )
    img = Image.open("PP.png")
    st.image(img)
    st.markdown(
        "<b><h1 style='text-align: #center; color:black ;font-size:25px ;'>▶ Columns in the dataset</h1>"
        , unsafe_allow_html=True   
        )
    st.markdown("**⟹ Species:** penguin species (Chinstrap, Adélie, or Gentoo)")
    st.markdown("**⟹ Island:** island name (Dream, Torgersen, or Biscoe) in the Palmer Archipelago (Antarctica)")
    st.markdown("**⟹ culmen_length_mm:** culmen length (mm)")
    st.markdown("**⟹ culmen_depth_mm:** culmen depth (mm)")
    st.markdown("**⟹flipper_length_mm:** flipper length (mm)")
    st.markdown("**⟹body_mass_g:** body mass (g)")
    st.markdown("**⟹Sex:** penguin sex")
    



    st.write("""
    Data obtained from the **[palmerpenguins library](https://github.com/allisonhorst/palmerpenguins)** in R by Allison Horst.
    """)

