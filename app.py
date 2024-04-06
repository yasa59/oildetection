import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os

# Set random seeds for reproducibility
np.random.seed(0)
tf.random.set_seed(0)

# Initialize data augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=20,  # Rotate images by up to 20 degrees
    width_shift_range=0.2,  # Shift images horizontally by up to 20% of the width
    height_shift_range=0.2,  # Shift images vertically by up to 20% of the height
    shear_range=0.2,  # Shear transformation with a max shear angle of 20 degrees
    zoom_range=0.2,  # Zoom into images by up to 20%
    horizontal_flip=True,  # Flip images horizontally
    vertical_flip=True,  # Flip images vertically
    brightness_range=[0.8, 1.2],  # Adjust brightness of images
    fill_mode='nearest'  # Fill in newly created pixels after transformations
)

# Load the pre-trained face oil detection model
oil_model = load_model('newmodel.h5')  # Ensure 'newmodel.h5' is available in your project directory

# Function to apply data augmentation and preprocess an image
def preprocess_image(face_img, target_size=(224, 224)):
    face_img = datagen.random_transform(face_img)  # Apply data augmentation
    processed_img = cv2.resize(face_img, target_size)  # Resize to target size
    processed_img = processed_img.astype('float32') / 255.0  # Normalize pixel values
    processed_img = np.expand_dims(processed_img, axis=0)  # Add batch dimension
    return processed_img

# Function to map model prediction to oiliness level
def map_to_level(oil_prediction):
    if oil_prediction < 0.25:
        return "Low level"
    elif oil_prediction < 0.5:
        return "Normal level"
    elif oil_prediction < 0.75:
        return "Middle level"
    else:
        return "High level"

# Function to predict skin type from an image
def predict_skin_type(image_path):
    input_image = cv2.imread(image_path)  # Read the input image
    processed_image = preprocess_image(input_image)  # Preprocess the image

    # Predict skin type using the pre-trained model
    oil_prediction = oil_model.predict(processed_image)
    
    skin_type = 'Oily' if oil_prediction[0][0] > 0.5 else 'Non-oily'  # Determine skin type
    percentage = oil_prediction[0][0]  # Oiliness percentage
    oiliness_level = map_to_level(oil_prediction[0][0])  # Map prediction to oiliness level
    
    return skin_type, percentage, oiliness_level

# Functions to recommend treatment and beauty products 
def recommend_treatment(oiliness_level):
    if oiliness_level == "Low level":
        return [
            "Use a gentle, hydrating cleanser that doesn't strip away natural oils.",
            "Moisturize regularly with a rich, creamy moisturizer to keep skin hydrated.",
            "Use products with ingredients like hyaluronic acid, glycerin, and ceramides to lock in moisture.",
            "Limit hot showers and baths, as hot water can further dry out the skin.",
            "Exfoliate gently to remove dead skin cells and promote cell turnover.",
            "Use a humidifier in dry indoor environments to add moisture to the air."
        ]
    elif oiliness_level == "Normal level":
        return [
            "Use a gentle cleanser suitable for your skin type to maintain balance.",
            "Moisturize regularly to keep skin hydrated and balanced.",
            "Use sunscreen daily to protect against UV damage and premature aging.",
            "Maintain a healthy diet and stay hydrated for overall skin health."
        ]
    elif oiliness_level == "Middle level":
        return [
            "Use a mild cleanser that doesn't overly dry out or irritate the skin.",
            "Use a lightweight, oil-free moisturizer on areas that tend to be dry.",
            "Use targeted treatments for specific skin concerns, such as acne or dry patches.",
            "Consider using a mattifying primer or oil-absorbing products on oily areas.",
            "Adjust your skincare routine based on how your skin feels in different areas."
        ]
    elif oiliness_level == "High level":
        return [
            "Use a gentle, foaming cleanser to remove excess oil and impurities.",
            "Use oil-free or mattifying moisturizers to hydrate without adding excess oil.",
            "Use products with ingredients like salicylic acid or benzoyl peroxide to control acne and breakouts.",
            "Use a clay mask or exfoliating treatment 1-2 times a week to help control oil production.",
            "Avoid heavy or greasy products that can clog pores and exacerbate oiliness."
        ]
    else:
        return ["Skin type not recognized. Please consult a dermatologist for personalized recommendations."]

def recommend_beauty_products(oiliness_level):
    if oiliness_level == "Low level":
        return [
            "Hydrating Cleanser: CeraVe Hydrating Facial Cleanser",
            "Rich Moisturizer: Cetaphil Rich Hydrating Night Cream",
            "Hydrating Serum: Neutrogena Hydro Boost Hydrating Serum"
        ]
    elif oiliness_level == "Normal level":
        return [
            "Gentle Cleanser: Cetaphil Gentle Skin Cleanser",
            "Moisturizer: Neutrogena Hydro Boost Water Gel",
            "Sunscreen: La Roche-Posay Anthelios Melt-in Milk Sunscreen"
        ]
    elif oiliness_level == "Middle level":
        return [
            "Mild Cleanser: Cetaphil Daily Facial Cleanser",
            "Lightweight Moisturizer: La Roche-Posay Toleriane Double Repair Face Moisturizer",
            "Targeted Treatments: The Ordinary Niacinamide 10% + Zinc 1% Serum for oily areas, The Ordinary Hyaluronic Acid 2% + B5 for dry areas"
        ]
    elif oiliness_level == "High level":
        return [
            "Foaming Cleanser: CeraVe Foaming Facial Cleanser",
            "Oil-Free Moisturizer: Paula's Choice Skin Balancing Invisible Finish Moisture Gel",
            "Acne Control Treatment: The Ordinary Salicylic Acid 2% Solution"
        ]
    else:
        return ["Skin type not recognized. Please consult a dermatologist for personalized recommendations."]

def app():
    st.title("Skin Type Detection and Recommendations")

    # File uploader allows user to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])
    if uploaded_file is not None:
        # Convert the uploaded file to an OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Display the uploaded image
        st.image(opencv_image, channels="BGR", caption="Uploaded Image")
        
        # Save the uploaded image to a temporary file
        temp_image_path = 'temp_image.jpg'
        cv2.imwrite(temp_image_path, opencv_image)

        # Predict the skin type based on the uploaded image
        skin_type, percentage, oiliness_level = predict_skin_type(temp_image_path)
        
        # Display prediction results
        st.write(f"Predicted Skin Type: {skin_type} | Oiliness Percentage: {percentage*100:.2f}% | Oiliness Level: {oiliness_level}")

        # Display treatment recommendations
        st.subheader("Treatment Recommendations")
        treatment_info = recommend_treatment(oiliness_level)
        for info in treatment_info:
            st.write("- " + info)

        # Display beauty product recommendations
        st.subheader("Beauty Products Recommendations")
        product_info = recommend_beauty_products(oiliness_level)
        for info in product_info:
            st.write("- " + info)

        # Clean up: Remove the temporary image file after processing
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

if __name__ == "__main__":
    app()

