import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# =========================
# LOAD MODEL
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_XCEPTION = os.path.join(BASE_DIR, "xception.h5")

model = tf.keras.models.load_model(MODEL_XCEPTION, compile=False)

print("✅ Xception model loaded successfully")

# =========================
# PLANT CLASS LABELS
# =========================

PLANT_CLASSES = {
    0: 'Aloevera',
    1: 'Neem',
    2: 'Tulasi'
}

# =========================
# PLANT INFORMATION
# =========================

PLANT_INFO = {

    "Aloevera": {

        "about": "Aloe Vera is a medicinal succulent plant widely used in herbal and skincare treatments.",

        "uses": [
            "Used for treating burns, cuts, and skin irritation.",
            "Used in herbal drinks for digestion improvement.",
            "Used in cosmetic and skincare products."
        ],

        "benefits": [
            "Boosts skin healing naturally.",
            "Helps reduce acne and skin inflammation.",
            "Improves digestion and gut health.",
            "Contains antioxidant and antibacterial properties.",
            "Supports immunity and hydration."
        ]
    },

    "Neem": {

        "about": "Neem is a powerful Ayurvedic medicinal plant known for its antibacterial properties.",

        "uses": [
            "Used for treating skin infections and acne.",
            "Used in Ayurvedic medicines and oils.",
            "Used in dental care and immunity treatments."
        ],

        "benefits": [
            "Purifies blood naturally.",
            "Boosts immunity and body protection.",
            "Helps control acne and skin diseases.",
            "Contains strong antibacterial properties.",
            "Supports healthy hair and scalp."
        ]
    },

    "Tulasi": {

        "about": "Tulasi is a sacred medicinal herb in Ayurveda known for its healing and immunity boosting properties.",

        "uses": [
            "Used in herbal tea and Ayurvedic medicine.",
            "Used for cough, cold, and respiratory relief.",
            "Used in immunity boosting home remedies."
        ],

        "benefits": [
            "Boosts immunity naturally.",
            "Helps reduce stress and anxiety.",
            "Improves respiratory health.",
            "Contains antibacterial and antiviral properties.",
            "Supports heart and digestive health."
        ]
    }
}

# =========================
# PREDICTION FUNCTION
# =========================

def predict_plant(img_path):

    # Load image
    img = load_img(img_path, target_size=(256, 256))

    # Convert image to array
    img = img_to_array(img)

    # Normalize image
    img = img / 255.0

    # Expand dimensions
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)

    # Get class index
    class_idx = np.argmax(prediction)

    # Confidence score
    confidence = float(np.max(prediction) * 100)

    # Plant name
    plant_name = PLANT_CLASSES.get(class_idx, "Unknown Plant")

    # Plant details
    plant_info = PLANT_INFO.get(plant_name, {
        "about": "Information not available.",
        "uses": [],
        "benefits": []
    })

    print("\n✅ Prediction Successful!")
    print(f"🌿 Predicted Plant: {plant_name}")
    print(f"📊 Confidence: {confidence:.2f}%\n")

    return {
        "plant_name": plant_name,
        "confidence": round(confidence, 2),
        "about": plant_info["about"],
        "uses": plant_info["uses"],
        "benefits": plant_info["benefits"]
    }