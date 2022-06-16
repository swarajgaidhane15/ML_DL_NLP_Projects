import os
import pickle

def get_filenames():
    filenames = []
    
    if 'filenames.pkl' not in os.listdir():
        players = os.listdir("images")

        for player in players:
            for file in os.listdir(os.path.join('images', player)):
                filenames.append(os.path.join('images', player, file))

        pickle.dump(filenames, open('filenames.pkl', 'wb'))

    else:
        filenames = pickle.load(open('filenames.pkl', 'rb'))
    
    return filenames
