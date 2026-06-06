import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt
import io
import os

#  Configuration de la page
st.set_page_config(
    page_title="Classificateur de Fleurs",
    page_icon="🌸",
    layout="centered"
)

#  Classes du modèle et paramètres 
CLASSES     = ["Rose", "Tulipe", "Tournesol", "Jonquille", "Pensée"]
IMG_SIZE    = (224, 224)
MODEL_PATH  = "modele_fleurs.h5"     # Fichier du modèle sauvegardé

#  Emojis par classe pour l'affichage
EMOJIS = {
    "Rose":       "🌹",
    "Tulipe":     "🌷",
    "Tournesol":  "🌻",
    "Jonquille":  "🌼",
    "Pensée":     "💜",
}


# Chargement du modèle (mis en cache pour ne charger qu'une seule fois)
@st.cache_resource
def charger_modele():
    """Charge le modèle Keras depuis le fichier .h5."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Fichier modèle introuvable : {MODEL_PATH}")
        st.info(
            "Pour générer le fichier modèle, exécutez d'abord :\n"
            "```python\nmodel.save('modele_fleurs.h5')\n```\n"
            "depuis votre notebook d'entraînement (checkpoint 18)."
        )
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)


def pretraiter_image(image: Image.Image) -> np.ndarray:
    """
    Prépare une image PIL pour l'inférence ResNet50.
    - Redimensionne à 224×224
    - Applique la normalisation ResNet50 [-1, 1]
    - Ajoute la dimension batch
    """
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    arr   = np.array(image, dtype=np.float32)
    arr   = preprocess_input(arr)       # Normalisation ResNet50
    arr   = np.expand_dims(arr, axis=0) # (1, 224, 224, 3)
    return arr


def creer_graphique_proba(probas: np.ndarray) -> plt.Figure:
    """Crée un graphique horizontal des probabilités par classe."""
    fig, ax = plt.subplots(figsize=(6, 3))
    colors  = ["#4CAF50" if p == max(probas) else "#90CAF9" for p in probas]
    ax.barh(CLASSES, probas * 100, color=colors)
    ax.set_xlabel("Probabilité (%)")
    ax.set_xlim([0, 100])
    ax.set_title("Distribution des probabilités")
    for i, (v, p) in enumerate(zip(probas, CLASSES)):
        ax.text(v * 100 + 1, i, f"{v*100:.1f}%", va='center', fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    return fig


# Interface Streamlit

#  En-tête 
st.title("🌸 Classificateur de Fleurs")
st.markdown(
    "Téléversez une photo de fleur pour identifier sa **classe**  \n"
    "parmi : **Rose · Tulipe · Tournesol · Jonquille · Pensée**"
)
st.divider()

#  Barre latérale : informations 
with st.sidebar:
    st.header("ℹ️ Informations")
    st.markdown("**Modèle :** ResNet50 (Transfer Learning)")
    st.markdown("**Dataset :** Oxford Flowers 102 (5 classes)")
    st.markdown("**Taille d'entrée :** 224 × 224 pixels")
    st.markdown("**Classes :**")
    for cls in CLASSES:
        st.markdown(f"  {EMOJIS[cls]} {cls}")
    st.divider()

#  Chargement du modèle 
with st.spinner("Chargement du modèle..."):
    model = charger_modele()
st.success("Modèle chargé avec succès !")

st.divider()

#  Zone de téléversement 
st.subheader("📤 Téléverser une image")
fichier = st.file_uploader(
    "Choisissez une image (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
    help="L'image sera redimensionnée en 224×224 pour le modèle."
)

#  Traitement et prédiction 
if fichier is not None:

    image = Image.open(fichier)

    # Affichage de l'image uploadée
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("🖼️ Image téléversée")
        st.image(image, use_container_width=True)
        st.caption(f"Taille originale : {image.size[0]} × {image.size[1]} px")

    # Prédiction
    with st.spinner("Analyse en cours..."):
        img_prep = pretraiter_image(image)
        probas   = model.predict(img_prep, verbose=0)[0]  # (5,)
        idx_pred = int(np.argmax(probas))
        classe   = CLASSES[idx_pred]
        confiance = float(probas[idx_pred]) * 100

    with col2:
        st.subheader("🎯 Résultat")
        st.markdown(
            f"<div style='background:#f0fff4;border-radius:10px;padding:20px;text-align:center'>"
            f"<h2>{EMOJIS[classe]}</h2>"
            f"<h3 style='color:#2e7d32'>{classe}</h3>"
            f"<p style='font-size:22px;color:#388e3c'><b>{confiance:.1f}%</b> de confiance</p>"
            f"</div>",
            unsafe_allow_html=True
        )

        # Niveau de confiance coloré
        if confiance >= 80:
            st.success(f"✅ Prédiction très fiable ({confiance:.1f}%)")
        elif confiance >= 55:
            st.warning(f"⚠️ Prédiction modérée ({confiance:.1f}%) — essayez une image plus nette")
        else:
            st.error(f"❌ Prédiction incertaine ({confiance:.1f}%) — image hors domaine ?")

    st.divider()

    # Graphique des probabilités
    st.subheader("📊 Probabilités par classe")
    fig = creer_graphique_proba(probas)
    st.pyplot(fig)

    # Tableau détaillé
    st.subheader("📋 Détail des scores")
    data = {
        "Classe": [f"{EMOJIS[c]} {c}" for c in CLASSES],
        "Probabilité": [f"{p*100:.2f}%" for p in probas],
        "Confiance": ["⭐" * min(5, int(p * 10)) for p in probas],
    }
    st.dataframe(data, use_container_width=True, hide_index=True)

    # Bouton de téléchargement des résultats
    st.divider()
    rapport = (
        f"RAPPORT DE PRÉDICTION — Classificateur de Fleurs\n"
        f"{'='*50}\n"
        f"Classe prédite  : {classe}\n"
        f"Confiance       : {confiance:.2f}%\n\n"
        f"Distribution des probabilités :\n"
        + "\n".join([f"  {CLASSES[i]:<12}: {probas[i]*100:.2f}%" for i in range(len(CLASSES))])
        + f"\n\nModèle : ResNet50 fine-tuné | Dataset : Oxford Flowers 5 classes"
    )
    st.download_button(
        label="📥 Télécharger le rapport",
        data=rapport,
        file_name="rapport_prediction.txt",
        mime="text/plain"
    )

else:
    # Placeholder quand aucune image n'est chargée
    st.info("👆 Téléversez une image ci-dessus pour obtenir une prédiction.")

    # Exemples de classes
    st.subheader("🌺 Classes reconnues par le modèle")
    cols = st.columns(len(CLASSES))
    for col, cls in zip(cols, CLASSES):
        col.markdown(
            f"<div style='text-align:center;padding:12px;background:#f8f9fa;"
            f"border-radius:8px;margin:2px'>"
            f"<div style='font-size:28px'>{EMOJIS[cls]}</div>"
            f"<div style='font-weight:600;margin-top:6px'>{cls}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
