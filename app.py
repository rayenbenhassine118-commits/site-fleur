import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Configuration de la page Streamlit
st.set_page_config(page_title="Reconnaissance de Fleurs", page_icon="🌸", layout="centered")

# 2. Chargement du modèle IA (Mis en cache pour éviter de le recharger à chaque clic)
@st.cache_resource
def charger_modele():
    # compile=False est indispensable pour éviter les conflits de version sur le Cloud
    return tf.keras.models.load_model('mon_modele_fleurs.keras', compile=False)

model = charger_modele()

# 3. Liste des classes de fleurs (⚠️ À modifier selon l'ordre exact de ton entraînement)
classes_fleurs = ['Marguerite', 'Pissenlit', 'Rose', 'Tournesol', 'Tulipe']

# 4. Interface Utilisateur (Design de la page)
st.title("🌸 Classification de Fleurs par IA")
st.write("Téléchargez la photo d'une fleur pour que l'Intelligence Artificielle l'analyse.")

# Bouton d'upload d'image
fichier_upload = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

# 5. Logique de traitement et prédiction
if fichier_upload is not None:
    # Ouverture et affichage de l'image sélectionnée par l'utilisateur
    image = Image.open(fichier_upload)
    st.image(image, caption="Image sélectionnée", use_container_width=True)
    
    with st.spinner("Analyse de l'image en cours par l'IA..."):
        # Redimensionnement de l'image selon ce qu'attend ton modèle (Ex: 180x180)
        # ⚠️ Modifie (180, 180) si ton modèle a été entraîné sur une autre taille (ex: 224, 224)
        image_redimensionnee = image.resize((180, 180)) 
        
        # Transformation de l'image en tableau de nombres pour Keras
        img_array = tf.keras.utils.img_to_array(image_redimensionnee)
        img_array = tf.expand_dims(img_array, 0)  # Ajout de la dimension de batch

        # Exécution de la prédiction
        predictions = model.predict(img_array)
        
        # Extraction du résultat
        index_prediction = np.argmax(predictions[0])
        nom_prediction = classes_fleurs[index_prediction]
        
        # Calcul du pourcentage de confiance de l'IA
        try:
            pourcentage_confiance = 100 * np.max(tf.nn.softmax(predictions[0]))
        except:
            pourcentage_confiance = 100 * np.max(predictions[0])

    # 6. Affichage final (Vos lignes de codes exactes)
    st.success(f"🔮 Résultat de l'IA : C'est une **{nom_prediction.upper()}** !")
    st.info(f"📈 Niveau de confiance : **{pourcentage_confiance:.2f}%**")