import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from tqdm import tqdm

#get all the image filenames with their respective directory and saving in a pickle file
def generate_filenames_pkl():
    actors = os.listdir('data_set')
    # print(actors)

    filenames = []
    for actor in actors:
        # print(actor)
        if actor != '.DS_Store':
            files = os.listdir('data_set/'+actor)
            for file in files:
                filenames.append('data_set/'+actor+'/'+file)
    
    pickle.dump(filenames, open('img_pickle_dump.pkl', 'wb'))

    generate_features('img_pickle_dump.pkl')

    # print(filenames[0:10])

def generate_features(filename):
    model = VGGFace(model = 'resnet50', include_top = False, input_shape = (224,224,3), pooling = 'avg')
    
    filenames = pickle.load(open(filename,'rb'))
    print(filenames[7283])
    print(filenames[7282])


    feature_results = []

    for file in tqdm(filenames):
        if file != '.DS_Store':
            feature_results.append(predict_eachimage(file, model))
    
    # for i in range(len(filenames)):
    #     feature_results.append(predict_eachimage(filenames[i], model))
    #     print(filenames[i])
    #     print(feature_results[i])
    
    pickle.dump(feature_results, open('features_pickle_dump.pkl', 'wb'))

def predict_eachimage(imgPath, model):
    # print(imgPath)
    img = image.load_img(imgPath,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array,axis=0)
    preprocessed_img = preprocess_input(expanded_img)

    result = model.predict(preprocessed_img).flatten()

    return result

generate_filenames_pkl()