from keras_vggface.utils import preprocess_input  # lets say, normalizes image array
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import os
import pickle
import cv2
from mtcnn import MTCNN  # face detector model 
import numpy as np

from getFilenames import get_filenames
from get_model import model

feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = get_filenames()

detector = MTCNN()

def save_uploaded_img(uploaded):
    try:
        with open(os.path.join('uploads', uploaded.name), 'wb') as f:
            f.write(uploaded.getbuffer())
        return True
    except Exception as e:
        return False

def extract_features(img_path, model):
    img = cv2.imread(img_path)

    results = detector.detect_faces(img)
    x, y, width, height = results[0]['box']

    face = img[y:y+height, x:x+width]

    image = Image.fromarray(face)
    face_array = np.asarray(image.resize((224,224))).astype('float32')

    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)

    result = model.predict(preprocessed_img).flatten()
    return result

def recommend(features):
    similarity = []

    for i in  range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1,-1), feature_list[i].reshape(1,-1))[0][0])

    max_similarity_index = sorted(list(enumerate(similarity)), reverse=True, key=lambda x:x[1])[0][0]
    
    return max_similarity_index

def get_name(path):
    name = path.split("/")[1]

    return " ".join([word[0].upper() + word[1:] for word in name.split("_")])


st.title("Which cricketer are you ?")

uploaded_img = st.file_uploader('Choose an image')

if uploaded_img is not None:

    # Save the image in directory
    if save_uploaded_img(uploaded_img):

        # Load image
        selected_image = Image.open(uploaded_img)

        # Extract features
        features = extract_features(os.path.join('uploads', uploaded_img.name), model)

        # Recommend the result
        idx = recommend(features)

        # Display the result
        col1, col2 = st.columns(2)

        with col1:
            st.header("Your uploaded image")
            st.image(selected_image, width=300)

        with col2:
            st.header("You look like..")
            st.image(filenames[idx], width=300)
            st.header(get_name(filenames[idx]))