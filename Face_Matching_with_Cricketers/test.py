from keras_vggface.utils import preprocess_input  # lets say, normalizes image array
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import cv2
from mtcnn import MTCNN  # face detector model 
from PIL import Image

from getFilenames import get_filenames
from get_model import model

feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = get_filenames()

detector = MTCNN()

# Load image, detect face and extract its features
sample_img = cv2.imread('samples/alastir.webp')

results = detector.detect_faces(sample_img)
x, y, width, height = results[0]['box']

face = sample_img[y:y+height, x:x+width]

image = Image.fromarray(face)
face_array = np.asarray(image.resize((224,224))).astype('float32')

expanded_img = np.expand_dims(face_array, axis=0)
preprocessed_img = preprocess_input(expanded_img)

result = model.predict(preprocessed_img).flatten()

# Find cosine distance of current image with the other features
similarity = []

for i in  range(len(feature_list)):
    similarity.append(cosine_similarity(result.reshape(1,-1), feature_list[i].reshape(1,-1))[0][0])

max_similarity_index = sorted(list(enumerate(similarity)), reverse=True, key=lambda x:x[1])[0][0]

# Recommend that image
recom_img = cv2.imread(filenames[max_similarity_index])
cv2.imshow('Recommended Image', recom_img)
cv2.waitKey(0)