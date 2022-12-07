from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow import expand_dims
import numpy as np
import os


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static\images"
model = load_model('model_tbc_regularizer.h5')

@app.route('/')
def beranda():
    return render_template('index.html')

@app.route('/tutorial')
def tutorial():
    return render_template('tutorial.html')

@app.route('/periksa', methods=['GET','POST'])
def periksa():
    if request.method == 'POST':
        if request.files: 
            image = request.files['image']
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(img_path)
            prediction = get_output(img_path)
            return render_template('periksa.html', uploaded_image=image.filename, prediction=prediction)
    return render_template('periksa.html')

def get_output(img_path):
    loaded_img = load_img(img_path, target_size=(124,124))
    img_array = img_to_array(loaded_img) / 255.0
    img_array = expand_dims(img_array,0)
    classes = model.predict(img_array)
    if classes[0] <= 0.5:
        return 'Normal'    
    else:
        return 'Tuberculosis'

@app.route('/display/<filename>')
def send_uploaded_image(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ =='__main__':
    app.run(port=12000, debug = True)
