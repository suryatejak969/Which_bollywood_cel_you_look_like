from flask import Flask, render_template, request, redirect, url_for, flash
import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from sklearn.metrics.pairwise import cosine_similarity
from mtcnn import MTCNN
from PIL import Image
import cv2
from werkzeug.utils import secure_filename
import glob
import shutil

UPLOAD_FOLDER = 'static/uploads/'
app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

PRED_FOLDER = 'static/predicts/'
app.config['PRED_FOLDER'] = PRED_FOLDER

filenames = pickle.load(open('img_pickle_dump.pkl','rb'))

features = np.array(pickle.load(open('features_pickle_dump.pkl','rb')))

model = VGGFace(model = 'resnet50', include_top = False, input_shape = (224,224,3), pooling = 'avg')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/', methods=['POST'])
def predict():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    filepath = 'static/uploads/' + filename
    print(filepath)
    index = extractor(filepath)

    pred_file = filenames[index]
    print(pred_file)

    for jpgfile in glob.iglob(pred_file):
        shutil.copy(jpgfile, 'static/predicts')

	#print('upload_image filename: ' + filename)
    flash('Seems you look like ' + pred_file.split('/')[-2].replace('_', ' '))
    return render_template('home.html', filename=pred_file.split('/')[-1])
    # return filename

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='predicts/' + filename), code=301)


def extractor(filepath):
    detector = MTCNN()

    img = cv2.imread(filepath)
    results = detector.detect_faces(img)

    x,y,width,height = results[0]['box']

    face = img[y:y+height,x:x+width]

    img_array = Image.fromarray(face)
    img_array = img_array.resize((224,224))

    face_array = np.asarray(img_array)

    face_array = face_array.astype('float32')

    expanded_img = np.expand_dims(face_array,axis=0)
    preprocessed_img = preprocess_input(expanded_img)

    result = model.predict(preprocessed_img).flatten()

    similarity = []

    for i in range(len(features)):
        similarity.append(cosine_similarity(result.reshape(1,-1),features[i].reshape(1,-1))[0][0])
    

    index_pos = sorted(list(enumerate(similarity)),reverse=True,key=lambda x:x[1])[0][0]

    return index_pos

if __name__ == '__main__':
    app.run(debug = True)