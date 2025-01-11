import streamlit as st
import numpy as np
import time
import plotly.graph_objects as go
import pickle
import re
import string

st.set_page_config(layout="wide")

st.subheader("Text Classification: Real or Fake")

with open('fake_news_model.pkl', 'rb') as lr_model_file:
    lr_model = pickle.load(lr_model_file)

with open('fake_news_model_rf.pkl', 'rb') as rf_model_file:
    rf_model = pickle.load(rf_model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def wordclean(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

def show_gauge_chart(metric_name, value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={'text': metric_name, 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "purple"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': "red"},
                {'range': [50, 75], 'color': "yellow"},
                {'range': [75, 100], 'color': "lightgreen"}],
            'threshold': {
                'line': {'color': "green", 'width': 3},
                'thickness': 0.7,
                'value': value * 100}
        }
    ))
    fig.update_layout(width=250, height=250)
    st.plotly_chart(fig, use_container_width=False)

st.sidebar.header("Select Algorithm")
algorithm = st.sidebar.radio("Choose Algorithm", ["Logistic Regression", "Random Forest"])

col1, col2 = st.columns([1.2, 2])

metrics = {
    "Logistic Regression": {
        "accuracy": 0.9228,
        "precision": 0.92,
        "recall": 0.92,
        "f1_score": 0.92
    },
    "Random Forest": {
        "accuracy": 0.86,
        "precision": 0.87,
        "recall": 0.86,
        "f1_score": 0.86
    }
}

with col1:
    # 2. Input : Texte pour détecter réel ou fake
    st.subheader("Enter Text for Real/Fake Detection")
    user_text = st.text_area("Input your text here:", "")

    # 3. Lancer l'expérience avec un seul bouton
    if st.button("Launch Experiment"):  # Un seul bouton
        if not user_text:
            st.warning("Please input some text for detection!")
        else:
            # Animation de chargement
            with st.spinner("Processing the text... Please wait."):
                time.sleep(2)  # Simulation du temps de traitement

            # Prétraitement du texte et prédiction
            cleaned_text = wordclean(user_text)
            input_vector = vectorizer.transform([cleaned_text])

            # Prédiction selon l'algorithme sélectionné
            if algorithm == "Logistic Regression":
                prediction = lr_model.predict(input_vector)[0]
            else:  # Random Forest
                prediction = rf_model.predict(input_vector)[0]

            # Affichage des résultats
            result = "Real" if prediction == 1 else "Fake"
            st.subheader("Text Detection")
            time.sleep(1)  # Petit délai pour l'effet visuel
            st.success(f"🎯 Your text is predicted as: **{result}**")

with col2:
    if user_text:
        # Récupération des métriques pour le modèle sélectionné
        model_metrics = metrics[algorithm]

        st.subheader("Results")
        cols = st.columns(2)  # Deux colonnes pour un alignement optimal
        with cols[0]:
            show_gauge_chart("Accuracy", model_metrics["accuracy"])
            show_gauge_chart("Precision", model_metrics["precision"])
        with cols[1]:
            show_gauge_chart("Recall", model_metrics["recall"])
            show_gauge_chart("F1 Score", model_metrics["f1_score"])
