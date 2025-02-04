
import streamlit as st
from transformers import pipeline

st.title("distilbert_imdb_model")

classifier = pipeline('text-classification', model='bert-base-uncased-sentiment-model')

text = st.text_area("Enter Your Tweet Here")

if st.button("Predict"):
         result = classifier(text)
         st.write("Prediction Result:", result)
