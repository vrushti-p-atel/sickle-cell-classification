import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import applications
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.svm import SVC
import pickle
import os

# Load the trained SVM model
@st.cache_resource
def load_model():
    with open('svm_model.pkl', 'rb') as f:
        clf = pickle.load(f)
    return clf

# Load the CNN model for feature extraction
@st.cache_resource
def load_cnn():
    base_model = applications.ResNet50(weights='imagenet', include_top=True)
    vector = base_model.get_layer("avg_pool").output
    model = tf.keras.Model(base_model.input, vector)
    return model

# Classes
classes = ["Circular", "Falciforme (Sickle)", "Outras"]

# Class explanations
class_explanations = {
    "Circular": "Indicates normal, healthy red blood cells with a round shape. This is the expected morphology for properly functioning erythrocytes.",
    "Falciforme (Sickle)": "Indicates sickle-shaped cells, characteristic of sickle cell anemia. These abnormal shapes can block blood flow, cause tissue damage, and lead to severe pain and complications.",
    "Outras": "Indicates other abnormal shapes or conditions. These may include various morphological abnormalities that require further medical evaluation and diagnostic testing."
}

# Title and Introduction
st.title("🩸 Erythrocyte Classification for Sickle Cell Disease Detection")
st.markdown("""
### About Sickle Cell Disease
Sickle cell disease is a genetic blood disorder that affects the shape and function of red blood cells.
Normally round and flexible, affected red blood cells become rigid and sickle-shaped, which can block
blood flow and cause tissue damage, pain, anemia, and other serious complications.

### How This Tool Helps
This AI-powered tool analyzes microscopic images of red blood cells (erythrocytes) to classify them into three categories:
- **Circular**: Normal, healthy cells
- **Falciforme**: Sickle-shaped cells (indicative of sickle cell disease)
- **Outras**: Other abnormalities requiring medical attention

Early detection through automated image analysis can aid in diagnosis and monitoring of sickle cell disease.
""")

st.divider()

# Demo section
st.header("🔬 Try the Demo with Sample Images")
st.write("Click on any sample image below to test the classification system:")

# Sample images from each class
sample_images = {
    "Circular": "train-test-erythrocytes/dataset/5-fold/round_1/train/circular/c0042.jpg",
    "Falciforme (Sickle)": "train-test-erythrocytes/dataset/5-fold/round_1/train/falciforme/e0048.jpg",
    "Outras": "train-test-erythrocytes/dataset/5-fold/round_1/train/outras/o0049.jpg"
}

col1, col2, col3 = st.columns(3)

selected_sample = None
with col1:
    st.markdown("**Normal Cells**")
    if st.button("Test Circular Sample", key="circular_btn"):
        selected_sample = sample_images["Circular"]
    if selected_sample == sample_images["Circular"]:
        st.image(selected_sample, caption="Circular Erythrocyte (Normal)", use_container_width=True)

with col2:
    st.markdown("**Sickle Cells**")
    if st.button("Test Falciforme Sample", key="falciforme_btn"):
        selected_sample = sample_images["Falciforme (Sickle)"]
    if selected_sample == sample_images["Falciforme (Sickle)"]:
        st.image(selected_sample, caption="Falciforme Erythrocyte (Sickle-shaped)", use_container_width=True)

with col3:
    st.markdown("**Other Abnormalities**")
    if st.button("Test Outras Sample", key="outras_btn"):
        selected_sample = sample_images["Outras"]
    if selected_sample == sample_images["Outras"]:
        st.image(selected_sample, caption="Outras Erythrocyte (Other abnormalities)", use_container_width=True)

st.divider()

# Upload section
st.header("📤 Upload Your Own Erythrocyte Image")
st.write("Upload a microscopic image of red blood cells for classification:")
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "png", "jpeg", "bmp"])

# Handle both uploaded files and selected samples
file_to_process = uploaded_file if uploaded_file is not None else selected_sample

if file_to_process is not None:
    st.divider()
    st.header("🔍 Classification Results")

    # Display the image
    col1, col2 = st.columns([1, 2])
    with col1:
        if uploaded_file is not None:
            st.image(uploaded_file, caption='Uploaded Erythrocyte Image', use_container_width=True)
        else:
            st.image(file_to_process, caption='Selected Sample Image', use_container_width=True)

    with col2:
        # Preprocess the image
        img = image.load_img(file_to_process, target_size=(224, 224))
        img_arr = image.img_to_array(img)
        img_arr_b = np.expand_dims(img_arr, axis=0)
        input_img = preprocess_input(img_arr_b)

        # Load models
        cnn_model = load_cnn()
        svm_model = load_model()

        # Extract features
        with st.spinner("Analyzing image..."):
            features = cnn_model.predict(input_img, verbose=0)
            features = features.ravel().reshape(1, -1)

            # Predict
            prediction = svm_model.predict(features)[0]
            confidence = svm_model.decision_function(features)

        # Display results
        predicted_class = classes[prediction]

        st.success(f"### 🏆 Predicted Class: **{predicted_class}**")

        # Class explanation
        st.info(f"**What this means:** {class_explanations[predicted_class]}")

        # Confidence scores
        st.write("### 📊 Model Confidence Scores")
        confidence_scores = {}
        for i, cls in enumerate(classes):
            if i < len(confidence[0]):
                confidence_scores[cls] = confidence[0][i]
            else:
                confidence_scores[cls] = 0.0

        # Display as progress bars
        for cls in classes:
            score = confidence_scores[cls]
            percentage = min(max((score + 1) * 50, 0), 100)  # Normalize to 0-100
            st.write(f"**{cls}:** {percentage:.1f}%")
            st.progress(percentage / 100)

        # Technical details
        with st.expander("🔧 Technical Details"):
            st.write("**Model Used:** ResNet50 (CNN) + SVM Classifier")
            st.write("**Image Size:** 224x224 pixels")
            st.write("**Preprocessing:** Keras ResNet50 preprocess_input")
            st.write("**Feature Extraction:** Global Average Pooling")
            st.write(f"**Raw Confidence Scores:** {confidence}")

        # Medical disclaimer
        st.warning("""
        ⚠️ **Medical Disclaimer:** This tool is for educational and research purposes only.
        It should not be used as a substitute for professional medical diagnosis.
        Always consult with qualified healthcare professionals for medical decisions.
        """)