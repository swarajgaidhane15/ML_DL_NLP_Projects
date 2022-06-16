import os
import pickle

if 'embeddings.pkl' in os.listdir():
    features = pickle.load(open('embeddings.pkl', 'rb'))

else:
    from tensorflow.keras.preprocessing import image  # handles operations related to image
    from keras_vggface.utils import preprocess_input  # lets say, normalizes image array
    import numpy as np
    from tqdm import tqdm # adds progress bar

    from getFilenames import get_filenames
    from get_model import model

    filenames = get_filenames()

    def feature_extractor(img_path, model):
        img = image.load_img(img_path, target_size=(224,224))
        img_array = image.img_to_array(img)
        expanded_img = np.expand_dims(img_array, axis=0)  # creates a batch since model works on batches, i.e., (224, 224, 3) -> (1, 224, 224, 3)
        preprocessed_img = preprocess_input(expanded_img)  # refer line 2

        result = model.predict(preprocessed_img).flatten()
        return result

    features = []

    for file in tqdm(filenames):
        features.append(feature_extractor(file, model))

    pickle.dump(features, open('embeddings.pkl', 'wb'))
