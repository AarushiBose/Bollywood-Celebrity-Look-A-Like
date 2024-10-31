# Required libraries
import os
import pickle
import numpy as np
import streamlit as st
import cv2
from PIL import Image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity

# Initialize face detector and model
detector = MTCNN()
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# Load precomputed feature list and filenames
feature_list = pickle.load(open('embedding.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Function to save uploaded image
def save_uploaded_image(uploaded_image):
    try:
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        with open(os.path.join('uploads', uploaded_image.name), 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except Exception as e:
        st.write(f"Error saving uploaded image: {e}")
        return False

# Function to extract features from the uploaded image
def extract_features(img_path, model, detector):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)

    # Check if a face was detected
    if not results:
        st.write("No face detected in the uploaded image.")
        return None

    x, y, width, height = results[0]['box']
    face = img[y:y + height, x:x + width]

    # Prepare the face image
    image = Image.fromarray(face)
    image = image.resize((224, 224))
    face_array = np.asarray(image).astype('float32')

    # Preprocess and extract features
    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result

# Function to find the most similar Bollywood twin
def recommend(feature_list, features):
    similarity = [cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0] for i in
                  range(len(feature_list))]
    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos

# Streamlit app
st.title('Who is your Bollywood Twin?')

uploaded_image = st.file_uploader('Choose an image')

if uploaded_image is not None:
    # Save the image in a directory
    if save_uploaded_image(uploaded_image):
        # Load the image for display
        display_image = Image.open(uploaded_image)

        # Extract the features
        features = extract_features(os.path.join('uploads', uploaded_image.name), model, detector)

        if features is not None:
            # Recommend the most similar Bollywood twin
            index_pos = recommend(feature_list, features)
            predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))
            predicted_actor_path = os.path.normpath(filenames[index_pos])

            # Display results
            col1, col2 = st.columns(2)  # Updated from beta_columns to columns

            try:
                with col1:
                    st.header('Your uploaded image')
                    st.image(display_image)
                with col2:
                    st.header(f"Your Bollywood Twin is {predicted_actor}")
                    st.image(predicted_actor_path, width=300)
            except Exception as e:
                st.write(f"Error displaying images: {e}")
