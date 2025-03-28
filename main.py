import streamlit as st
from strm_moia import app as model1_app
from strm_2 import app as model2_app

st.set_page_config(page_title="Классификация спутниковых изображений", layout="wide")

st.sidebar.title("Выбор страницы")
page = st.sidebar.selectbox("Выберите модель", ["🌍 Моя модель (ResNet18)", "🧠 Pretrained ResNet & DenseNet"])

if page == "🌍 Моя модель (ResNet18)":
    model1_app()
elif page == "🧠 Pretrained ResNet & DenseNet":
    model2_app()
