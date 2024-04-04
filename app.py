from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image, ImageOps
import os
from mtcnn import MTCNN
import numpy as np
import requests
from io import BytesIO

# Initialize MTCNN detector and VGGFace model
detector = MTCNN()
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# Load precomputed embeddings and filenames
feature_list = pickle.load(open('embedding.pkl', 'rb'))
filenames = pickle.load(open('filenames_s3.pkl', 'rb'))

# Streamlit app title
st.title('Which Bollywood Celebrity are you?')

def save_uploaded_image(uploaded_image):
    try:
        with open(os.path.join('uploads', uploaded_image.name), 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except Exception as e:
        st.error(f"Error saving image: {e}")
        return False

def extract_features(img_path, model, detector):
    # Load the image with PIL
    img = Image.open(img_path).convert('RGB')
    img_array = np.asarray(img)

    # Detect faces
    results = detector.detect_faces(img_array)
    if results:
        x, y, width, height = results[0]['box']
        face = img_array[y:y+height, x:x+width]

        # Convert the face to PIL Image and resize
        image = Image.fromarray(face)
        image = ImageOps.fit(image, (224, 224))

        # Preprocess the face for the model
        face_array = np.asarray(image, dtype='float32')
        expanded_img = np.expand_dims(face_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img)

        # Predict features
        result = model.predict(preprocessed_img).flatten()
        return result
    else:
        st.warning("No face detected in the image.")
        return None


def recommend(feature_list, features):
    if features is not None:
        similarity = [cosine_similarity(features.reshape(1, -1), feat.reshape(1, -1))[0][0] for feat in feature_list]
        index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
        return index_pos

uploaded_image = st.file_uploader("Choose an image...")

if uploaded_image is not None:
    if save_uploaded_image(uploaded_image):
        display_image = Image.open(uploaded_image)
        st.image(display_image, caption='Uploaded Image.', use_column_width=True)

        features = extract_features(os.path.join('uploads', uploaded_image.name), model, detector)
        if features is not None:
            index_pos = recommend(feature_list, features)
            predicted_actor = " ".join(filenames[index_pos].split('/')[-1].split('_'))  # Assuming filenames now contain URLs

            # Display uploaded image
            col1, col2 = st.columns(2)
            with col1:
                st.header('Uploaded Image')
                st.image(display_image, caption='Uploaded Image.', use_column_width=True)

            # Fetch and display actor image from S3
            with col2:
                st.header(f"Looks like {predicted_actor} to me!")
                
                # Fetch the image from the URL
                actor_image_url = filenames[index_pos]  # This should be a URL
                response = requests.get(actor_image_url)
                actor_image = Image.open(BytesIO(response.content))

                st.image(actor_image, caption='Similar Celebrity.', use_column_width=True)
