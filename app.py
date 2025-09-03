import streamlit as st
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import time

# Load the trained model with error handling and caching
MODEL_PATH = "models/best_model.keras"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
    st.sidebar.success("✅ Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Error loading model: {str(e)}")
    st.error("Failed to load the AI model. Please check the model file and TensorFlow version.")
    st.stop()

# Class names
class_names = ['WithMask', 'WithoutMask']

# Function to preprocess the image
def preprocess_image(img):
    # Convert RGBA to RGB if needed
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .mask-on {
        background-color: #d4edda;
        border: 2px solid #c3e6cb;
    }
    .mask-off {
        background-color: #f8d7da;
        border: 2px solid #f5c6cb;
    }
    .confidence-bar {
        margin-top: 1rem;
    }
    .sidebar-content {
        padding: 1rem;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #666;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🛡 Face Mask Detection")
    st.markdown("---")
    st.markdown("### 📋 How to Use")
    st.markdown("1. Upload an image of a person's face")
    st.markdown("2. Wait for the AI to analyze")
    st.markdown("3. View the prediction and confidence")
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown("- *Framework:* TensorFlow/Keras")
    st.markdown("- *Input Size:* 224x224 pixels")
    st.markdown("- *Classes:* With Mask / Without Mask")
    st.markdown("---")
    st.markdown("### 🔧 Settings")
    show_confidence = st.checkbox("Show Confidence Bar", value=True)
    show_details = st.checkbox("Show Detailed Results", value=True)

# Main content
st.markdown('<h1 class="main-header">🛡 Face Mask Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced AI-powered mask detection for public safety</p>', unsafe_allow_html=True)

# File uploader with better styling
col1, col2, col3 = st.columns([1,2,1])
with col2:
    uploaded_file = st.file_uploader(
        "📤 Choose an image to analyze...",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear face image for best results"
    )

if uploaded_file is not None:
    # Create two columns for image and results
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📷 Uploaded Image")
        # Display the uploaded image
        image_pil = Image.open(uploaded_file)
        st.image(image_pil, caption='Uploaded Image', use_container_width=True)

    with col2:
        st.markdown("### 🔍 Analysis Results")

        # Progress bar for processing
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(100):
            progress_bar.progress(i + 1)
            status_text.text(f"Analyzing image... {i+1}%")
            time.sleep(0.01)

        status_text.text("Analysis complete!")

        # Preprocess the image
        processed_image = preprocess_image(image_pil)

        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)

        # Clear progress
        progress_bar.empty()
        status_text.empty()

        # Display results with custom styling
        if predicted_class == 0:
            st.markdown(f"""
            <div class="result-box mask-on">
                <h3 style="color: #155724;">✅ Mask Detected</h3>
                <p style="font-size: 1.2rem; color: #155724;">The person is wearing a mask properly!</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box mask-off">
                <h3 style="color: #721c24;">❌ No Mask Detected</h3>
                <p style="font-size: 1.2rem; color: #721c24;">The person is not wearing a mask. Please advise them to wear one.</p>
            </div>
            """, unsafe_allow_html=True)

        # Confidence display
        if show_confidence:
            st.markdown("### 📊 Confidence Level")
            confidence_percentage = float(confidence) * 100
            st.progress(float(confidence))
            st.markdown(f"{confidence_percentage:.1f}%** confidence in prediction")

        # Detailed results
        if show_details:
            st.markdown("### 📈 Detailed Analysis")
            with st.expander("View Prediction Details"):
                st.write(f"*Predicted Class:* {class_names[predicted_class]}")
                st.write(f"*Raw Confidence:* {confidence:.4f}")
                st.write("*Class Probabilities:*")
                for i, prob in enumerate(predictions[0]):
                    st.write(f"- {class_names[i]}: {prob:.4f} ({prob*100:.1f}%)")

# Footer
st.markdown("---")
st.markdown('<div class="footer">Built using Streamlit and TensorFlow | AI for Public Health</div>', unsafe_allow_html=True)