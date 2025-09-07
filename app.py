import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import applications
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.svm import SVC
import pickle
import os

# Configure page
st.set_page_config(
    page_title="Erythrocyte Classification",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .result-box {
        background-color: #f0f8e7;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained SVM model
@st.cache_resource
def load_model():
    try:
        with open('svm_model.pkl', 'rb') as f:
            clf = pickle.load(f)
        return clf
    except FileNotFoundError:
        st.error("❌ SVM model file not found. Please ensure 'svm_model.pkl' exists.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading SVM model: {str(e)}")
        return None

# Load the CNN model for feature extraction
@st.cache_resource
def load_cnn():
    try:
        base_model = applications.ResNet50(weights='imagenet', include_top=True)
        vector = base_model.get_layer("avg_pool").output
        model = tf.keras.Model(base_model.input, vector)
        return model
    except Exception as e:
        st.error(f"❌ Error loading ResNet50 model: {str(e)}")
        return None

# Classes
classes = ["Circular", "Falciforme (Sickle)", "Outras"]

# Class explanations
class_explanations = {
    "Circular": "Indicates normal, healthy red blood cells with a round shape. This is the expected morphology for properly functioning erythrocytes.",
    "Falciforme (Sickle)": "Indicates sickle-shaped cells, characteristic of sickle cell anemia. These abnormal shapes can block blood flow, cause tissue damage, and lead to severe pain and complications.",
    "Outras": "Indicates other abnormal shapes or conditions. These may include various morphological abnormalities that require further medical evaluation and diagnostic testing."
}

# Sidebar
with st.sidebar:
    st.title("🔧 Settings & Info")

    st.markdown("---")
    st.markdown("### 📊 Model Information")
    st.info("""
    **Architecture:** ResNet50 + SVM
    **Classes:** 3 (Circular, Falciforme, Outras)
    **Input Size:** 224x224 pixels
    **Training Data:** Erythrocyte microscopy images
    """)

    st.markdown("---")
    st.markdown("### 🎯 How to Use")
    st.markdown("""
    1. **Demo:** Click sample images to test
    2. **Upload:** Use file uploader for your images
    3. **Results:** View classification and confidence
    4. **Clear:** Reset to start over
    """)

    st.markdown("---")
    if st.button("🔄 Refresh App", help="Refresh the application"):
        st.rerun()

# Main content
st.title("🩸 Erythrocyte Classification for Sickle Cell Disease Detection")

# Create an info box for the introduction
st.info("""
**About Sickle Cell Disease:** A genetic blood disorder affecting red blood cell shape and function.
Normally round and flexible, affected cells become rigid and sickle-shaped, blocking blood flow
and causing pain, anemia, and complications.

**How This Tool Helps:** AI-powered analysis of microscopic erythrocyte images classifies cells into:
- **🟢 Circular**: Normal, healthy red blood cells
- **🔴 Falciforme**: Sickle-shaped cells (sickle cell disease indicator)
- **🟡 Outras**: Other abnormalities requiring medical evaluation

Early automated detection aids diagnosis and monitoring.
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

# Initialize session state for selected sample
if 'selected_sample' not in st.session_state:
    st.session_state.selected_sample = None

with col1:
    st.markdown("**🟢 Normal Cells**")
    st.image(sample_images["Circular"], caption="Circular Erythrocyte (Normal)", width='stretch')
    if st.button("Test This Sample", key="circular_btn"):
        st.session_state.selected_sample = sample_images["Circular"]

with col2:
    st.markdown("**🔴 Sickle Cells**")
    st.image(sample_images["Falciforme (Sickle)"], caption="Falciforme Erythrocyte (Sickle-shaped)", width='stretch')
    if st.button("Test This Sample", key="falciforme_btn"):
        st.session_state.selected_sample = sample_images["Falciforme (Sickle)"]

with col3:
    st.markdown("**🟡 Other Abnormalities**")
    st.image(sample_images["Outras"], caption="Outras Erythrocyte (Other abnormalities)", width='stretch')
    if st.button("Test This Sample", key="outras_btn"):
        st.session_state.selected_sample = sample_images["Outras"]

selected_sample = st.session_state.selected_sample

st.divider()

# Upload section
st.header("📤 Upload Your Own Erythrocyte Image")
st.markdown("""
Upload a microscopic image of red blood cells for instant classification.
Supported formats: JPG, PNG, JPEG, BMP
""")

uploaded_file = st.file_uploader(
    "Choose an image file...",
    type=["jpg", "png", "jpeg", "bmp"],
    help="Select a clear microscopic image of erythrocytes for analysis"
)

# Clear selection when uploading new file
if uploaded_file is not None:
    st.session_state.selected_sample = None

# Handle both uploaded files and selected samples
file_to_process = None
if uploaded_file is not None:
    file_to_process = uploaded_file
elif selected_sample is not None:
    file_to_process = selected_sample

if file_to_process is not None:
    st.divider()

    # Add a clear/reset button
    col_clear, col_title = st.columns([1, 4])
    with col_clear:
        if st.button("🔄 Clear & Start Over", help="Clear current selection and start fresh"):
            st.session_state.selected_sample = None
            st.rerun()
    with col_title:
        st.header("🔍 Classification Results")

    # Display the image
    col1, col2 = st.columns([1, 2])
    with col1:
        if uploaded_file is not None:
            st.image(uploaded_file, caption='Uploaded Erythrocyte Image', width='stretch')
        else:
            st.image(file_to_process, caption='Selected Sample Image', width='stretch')

    with col2:
        try:
            # Preprocess the image
            img = image.load_img(file_to_process, target_size=(224, 224))
            img_arr = image.img_to_array(img)
            img_arr_b = np.expand_dims(img_arr, axis=0)
            input_img = preprocess_input(img_arr_b)

            # Load models
            cnn_model = load_cnn()
            svm_model = load_model()

            if cnn_model is None or svm_model is None:
                st.error("❌ Failed to load models. Please check the model files.")
                st.stop()

            # Extract features
            with st.spinner("🔬 Analyzing image..."):
                features = cnn_model.predict(input_img, verbose=0)
                features = features.ravel().reshape(1, -1)

                # Predict
                prediction = svm_model.predict(features)[0]
                confidence = svm_model.decision_function(features)

            # Display results
            predicted_class = classes[prediction]

            st.success(f"### 🏆 Predicted Class: **{predicted_class}**")

            # Class explanation
            st.info(f"**📋 What this means:** {class_explanations[predicted_class]}")

            # Confidence scores
            st.write("### 📊 Model Confidence Scores")

            # Handle confidence scores properly for multi-class SVM
            if confidence.ndim > 1:
                confidence_scores = confidence[0]
            else:
                # For binary classification, we need to handle differently
                confidence_scores = np.zeros(len(classes))
                if len(classes) == 3:
                    # Map binary decision to multi-class interpretation
                    confidence_scores[prediction] = abs(confidence[0])
                    # Distribute remaining confidence
                    remaining_conf = 1.0 - confidence_scores[prediction]
                    for i in range(len(classes)):
                        if i != prediction:
                            confidence_scores[i] = remaining_conf / (len(classes) - 1)

            # Display as progress bars with better formatting
            for i, cls in enumerate(classes):
                score = confidence_scores[i] if i < len(confidence_scores) else 0.0
                # Normalize to 0-100 range for display
                percentage = min(max(score * 100, 0), 100)
                color = "🟢" if cls == predicted_class else "⚪"
                st.write(f"{color} **{cls}:** {percentage:.1f}% confidence")
                st.progress(percentage / 100)

        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            st.info("Please ensure the image is a valid JPG, PNG, or BMP file and try again.")

        # Technical details
        with st.expander("🔧 Technical Details"):
            st.write("**Model Used:** ResNet50 (CNN) + SVM Classifier")
            st.write("**Image Size:** 224x224 pixels")
            st.write("**Preprocessing:** Keras ResNet50 preprocess_input")
            st.write("**Feature Extraction:** Global Average Pooling")
            st.write(f"**Raw Confidence Scores:** {confidence}")

        # Medical disclaimer - make it more prominent
        st.error("""
        ### ⚠️ **Important Medical Disclaimer**
        **This tool is for educational and research purposes only.**
        It should **NOT** be used as a substitute for professional medical diagnosis or treatment.
        Always consult with qualified healthcare professionals for medical decisions and interpretations.
        Results should be validated by trained medical personnel.
        """)