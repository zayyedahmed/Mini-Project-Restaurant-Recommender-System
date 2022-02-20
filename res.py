from unittest import result
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

st.title("Restaurant Recommender System in Mexico")
resto_profile = pd.read_csv("resto_profile.csv")
resto_profile['desc'] = resto_profile['city'].str.cat(
    resto_profile[['alcohol', 'smoking_area', 'dress_code', 'price', 'Rambience', 'area']],
    sep = ' '
)
CV = CountVectorizer(
    tokenizer = lambda i: i.split(' '),
    analyzer = 'word'
)
matrix = CV.fit_transform(resto_profile['desc'])
type_desc = CV.get_feature_names()
s_desc = len(type_desc)
e_desc = matrix.toarray()
score = cosine_similarity(matrix)
X1 = resto_profile.drop(
    ['placeID', 'latitude', 'longitude', 'name', 'address', 'state', 'country', 'accessibility',
    'franchise', 'other_services', 'desc'], axis = 1)
y1 = resto_profile['name']

cit = st.selectbox("City", X1.city.unique())
alc = st.selectbox("Alcohol", X1.alcohol.unique())
smok = st.selectbox("Smoking Area", X1.smoking_area.unique())
dc = st.selectbox("Dress Code", X1.dress_code.unique())
pric = st.selectbox("Price", X1.price.unique())
ram = st.selectbox("Rambience", X1.Rambience.unique())
ar = st.selectbox("Area", X1.area.unique())

pred = pd.DataFrame([{
    'city': cit,
    'alcohol': alc,
    'smoking_area': smok,
    'dress_code': dc,
    'price' : pric,
    'Rambience': ram,
    'area': ar
}], index=[0])

model = joblib.load('model_rfc.sav')
if st.button("Recommended"):
    result = model.predict(pred)[0]
    index_res = resto_profile[resto_profile['name'] == result].index.values[0]
    resto = list(enumerate(score[index_res]))
    similar = sorted(
    resto,
    key = lambda i: i[1],
    reverse = True)
    recom = []
    for i in similar:
        if i[1] > 0.7:
            recom.append(i)
        else:
            pass

    import random
    rek = random.choices(recom, k = 5)
    empty_list = []
    for i in rek:
        reccom = {}
        j = 0
        while j < 8:
            reccom['name'] = resto_profile.iloc[i[0]]['name'].title(),
            reccom['city'] = resto_profile.iloc[i[0]]['city'],
            reccom['alcohol'] = resto_profile.iloc[i[0]]['alcohol'],
            reccom['smoking_area'] = resto_profile.iloc[i[0]]['smoking_area'],
            reccom['dress_code'] = resto_profile.iloc[i[0]]['dress_code'],
            reccom['price'] = resto_profile.iloc[i[0]]['price'],
            reccom['Rambience'] = resto_profile.iloc[i[0]]['Rambience'],
            reccom['area'] = resto_profile.iloc[i[0]]['area']
            reccom['latitude'] = resto_profile.iloc[i[0]]['latitude']
            reccom['longitude'] = resto_profile.iloc[i[0]]['longitude']
            j += 1
        empty_list.append(reccom)
    uu = pd.DataFrame(empty_list)
    st.write(uu)
    ii = uu[["name", "latitude", "longitude"]]
    st.map(ii)

