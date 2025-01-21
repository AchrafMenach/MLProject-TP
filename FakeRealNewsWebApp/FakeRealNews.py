import streamlit as st
import numpy as np
import time
import plotly.graph_objects as go
import pickle
import re
import string
import os
import nltk
from nltk.corpus import stopwords

# Téléchargement des stopwords si nécessaire
nltk.download('stopwords')

# Configuration de la page
st.set_page_config(layout="wide")

# Titre de l'application
st.subheader("Text Classification: Real or Fake")

# Chargement des fichiers de modèles et du vectoriseur
def load_model(file_path, model_name):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    else:
        st.error(f"{model_name} file not found.")
        return None

lr_model = load_model('fake_news_model.pkl', 'Logistic Regression Model')
rf_model = load_model('fake_news_model_rf.pkl', 'Random Forest Model')
vectorizer = load_model('vectorizer.pkl', 'Vectorizer')

# Vérification que les modèles et le vectoriseur sont bien chargés
if not lr_model or not rf_model or not vectorizer:
    st.stop()

# Fonction de nettoyage de texte
def wordclean(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Supprimer les URLs
    text = re.sub(r'[^\w\s]', '', text)  # Supprimer la ponctuation
    text = re.sub(r'\d+', '', text)  # Supprimer les chiffres
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]  # Supprimer les stopwords
    return ' '.join(words)

# Fonction d'affichage des jauges
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
                {'range': [75, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "green", 'width': 3},
                'thickness': 0.7,
                'value': value * 100
            }
        }
    ))
    fig.update_layout(width=250, height=250)
    st.plotly_chart(fig, use_container_width=False)

# Sidebar pour sélectionner l'algorithme
st.sidebar.header("Select Algorithm")
algorithm = st.sidebar.radio("Choose Algorithm", ["Logistic Regression", "Random Forest"])

# Colonnes pour la mise en page
col1, col2 = st.columns([1.2, 2])

# Dictionnaire des métriques
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

# Entrée utilisateur et détection
with col1:
    st.subheader("Enter Text for Real/Fake Detection")
    user_text = st.text_area("Input your text here:", "")

    if st.button("Launch Experiment"):
        if not user_text:
            st.warning("Please input some text for detection!")
        else:
            with st.spinner("Processing the text... Please wait."):
                time.sleep(2)

            # Nettoyage du texte et vectorisation
            try:
                cleaned_text = wordclean(user_text)
                input_vector = vectorizer.transform([cleaned_text])

                # Prédiction
                if algorithm == "Logistic Regression":
                    prediction = lr_model.predict(input_vector)[0]
                else:
                    prediction = rf_model.predict(input_vector)[0]

                # Affichage du résultat
                result = "Real" if prediction == 1 else "Fake"
                st.subheader("Text Detection")
                st.success(f"🎯 Your text is predicted as: **{result}**")
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Affichage des métriques
with col2:
    if user_text:
        model_metrics = metrics[algorithm]
        st.subheader("Results")
        cols = st.columns(2)
        with cols[0]:
            show_gauge_chart("Accuracy", model_metrics["accuracy"])
            show_gauge_chart("Precision", model_metrics["precision"])
        with cols[1]:
            show_gauge_chart("Recall", model_metrics["recall"])
            show_gauge_chart("F1 Score", model_metrics["f1_score"])
