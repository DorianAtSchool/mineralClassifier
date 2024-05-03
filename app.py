import os
import numpy as np
import cv2
import keras
from flask import Flask, render_template, request
import cv2
import numpy as np
from werkzeug.utils import secure_filename


#load model
model = keras.models.load_model('mineral_classification.h5')

app = Flask(__name__)
file_path = '0002.jpg'

@app.route('/')
def home():
    return render_template('upload_form.html', user_image = file_path, pred = "")

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if 'file' in request.files:
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Here you should save the file in static folder
            file.save(os.path.join('static', filename))
            pred = modelPrediction(filename)
            #file.save(os.path.join('static', filename))
            return render_template('upload_form.html', user_image = filename, pred = pred)

    return render_template('upload_form.html', user_image = file_path, pred = "Could Not Detect")

ALLOWED_EXTENSIONS = {'jpg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def modelPrediction(filename): 

    class_dict = {0: 'biotite',
    1: 'bornite',
    2:'chrysocolla',
    3: 'malachite',
    4: 'muscovite',
    5: 'pyrite',
    6: 'quartz'}

    file_path =  os.path.join("static", filename)
    test_image = cv2.imread(file_path)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    test_image = cv2.resize(test_image, (224,224),interpolation=cv2.INTER_CUBIC)
    #plt.imshow(test_image)
    test_image = np.expand_dims(test_image,axis=0)
    probs = model.predict(test_image)
    pred_class = np.argmax(probs)
    pred_class = class_dict[pred_class]
    return pred_class
    
app.run()
