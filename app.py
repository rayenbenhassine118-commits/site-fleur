import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Configuration de la page web
st.set_page_config(page_title="Détecteur de Fleurs IA", page_icon="🌸")
st.title("🌸 Mon Premier Détecteur de Fleurs IA")
st.write("Télécharge une photo de fleur pour que l'IA l'analyse en direct depuis ton PC !")

# 2. Chargement de ton modèle de neurones
@st.cache_resource
def charger_modele():
    return tf.keras.models.load_model('mon_modele_fleurs.keras')

with st.spinner("Lancement de l'IA en cours..."):
    model = charger_modele()

# Liste des fleurs dans l'ordre exact de ton entraînement Colab
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

# 3. Zone d'envoi de l'image sur l'interface du site
uploaded_file = st.file_uploader("Choisis une image de fleur...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Affichage de l'image
    image = Image.open(uploaded_file)
    st.image(image, caption='Ton image téléchargée', use_container_width=True)
    
    with st.spinner("L'IA analyse l'image..."):
        # Redimensionnement automatique à la taille requise par ton réseau (180x180)
        img_resized = image.resize((180, 180))
        img_array = tf.keras.utils.img_to_array(img_resized)
        img_array = tf.expand_dims(img_array, 0) # Formatage pour le modèle

        # Calcul de la prédiction
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        nom_prediction = class_names[np.argmax(score)]
        pourcentage_confiance = 100 * np.max(score)

    # 4. Affichage du résultat final en vert
    # 4. Affichage du résultat final en vert
    st.success(f"🔮 Résultat de l'IA : C'est une **{nom_prediction.upper()}** !")
    st.info(f"📈 Niveau de confiance : **{pourcentage_confiance:.2f}%**")