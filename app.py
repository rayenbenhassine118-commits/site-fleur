import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Configuration de la page
st.set_page_config(page_title="Reconnaissance de Fleurs", page_icon="🌸", layout="centered")

# 2. Chargement du modèle (mis en cache pour la rapidité, avec compile=False pour éviter l'erreur)
@st.cache_resource
def charger_modele():
    return tf.keras.models.load_model('mon_modele_fleurs.keras', compile=False)

model = charger_modele()

# 3. Liste des noms de fleurs (⚠️ Modifie cette liste selon les classes de ton modèle)
classes_fleurs = ['Marguerite', 'Pissenlit', 'Rose', 'Tournesol', 'Tulipe']

# 4. Interface Utilisateur
st.title("🌸 Classification de Fleurs par IA")
st.write("Téléchargez l'image d'une fleur, et notre modèle d'Intelligence Artificielle vous dira de quelle espèce il s'agit !")

# Zone de téléchargement d'image
fichier_upload = st.file_uploader("Choisissez une image de fleur (jpg, jpeg, png)...", type=["jpg", "jpeg", "png"])

# 5. Traitement de l'image et Prédiction
if fichier_upload is not None:
    # Affichage de l'image téléchargée
    image = Image.open(fichier_upload)
    st.image(image, caption="Image téléchargée", use_container_width=True)
    
    with st.spinner("L'IA analyse l'image..."):
        # Redimensionner l'image pour qu'elle corresponde à ce qu'attend le modèle (souvent 180x180 ou 224x224)
        # ⚠️ Modifie (180, 180) par la taille que tu as utilisée pour entraîner ton modèle
        image_redimensionnee = image.resize((180, 180)) 
        
        # Convertir l'image en tableau numpy et préparer le format pour Keras
        img_array = tf.keras.utils.img_to_array(image_redimensionnee)
        img_array = tf.expand_dims(img_array, 0) # Créer un batch d'une seule image

        # Faire la prédiction
        predictions = model.predict(img_array)
        
        # Calcul du résultat (en supposant que ton modèle utilise softmax)
        index_prediction = np.argmax(predictions[0])
        nom_prediction = classes_fleurs[index_prediction]
        
        # Si le modèle retourne des probabilités brutes, on utilise softmax, sinon on prend juste le max
        try:
            pourcentage_confiance = 100 * np.max(tf.nn.softmax(predictions[0]))
        except:
            pourcentage_confiance = 100 * np.max(predictions[0])

    # 6. Affichage du résultat final avec tes propres lignes de code
    st.success(f"🔮 Résultat de l'IA : C'est une **{nom_prediction.upper()}** !")
    st.info(f"📈 Niveau de confiance : **{pourcentage_confiance:.2f}%**")