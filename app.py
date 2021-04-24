from flask import Flask, render_template, request, flash, redirect
import torch
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from skimage.transform import resize
from neural_nets import *
import os
from torchvision.utils import save_image
from torchvision.transforms import transforms
from torch.autograd import Variable
import random
import string


app = Flask(__name__)


@app.route("/")
def home():
    return render_template('home.html')

@app.route("/pneumonia", methods=['GET', 'POST'])
def pneumoniaPage():
    return render_template('pneumonia.html')

@app.route("/mole", methods=['GET', 'POST'])
def molePage():
    return render_template('mole.html')

@app.route("/malariapredict", methods = ['POST', 'GET'])
def malariapredictPage():
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image'])
                img = img.resize((36,36))
                img = np.asarray(img)
                img = img.reshape((1,36,36,3))
                img = img.astype(np.float64)
                model = load_model("models/malaria.h5")
                pred = np.argmax(model.predict(img)[0])
        except:
            message = "Please upload an Image"
            return render_template('malaria.html', message = message)
    return render_template('malaria_predict.html', pred = pred)

@app.route("/pneumoniapredict", methods = ['POST', 'GET'])
def pneumoniapredictPage():
    if 'img_pneumonia' in request.files:
        img = Image.open(request.files['img_pneumonia']).convert('L')
        img = img.resize((36,36))
        img = np.asarray(img)
        img = img.reshape((1,36,36,1))
        img = img / 255.0
        model = load_model("models/pneumonia.h5")
        pred = np.argmax(model.predict(img)[0])
        return render_template('pneumonia_predict.html', pred = pred)
    else:
        message = "Please upload an Image"
    return render_template('pneumonia.html', message = message)

@app.route("/molesegmentation", methods = ['POST', 'GET'])
def molesegmentationPage():
    image = request.files.get('img_mole')
    if  not image:
        message = "Please upload an Image"
    else:
        img_init = Image.open(image).convert('RGB')
        transformation = transforms.Compose([transforms.ToTensor()])
        image_tensor = transformation(img_init).float()
        letters = string.ascii_lowercase
        src_input = 'static/images/'+'input'+''.join(random.choice(letters) for i in range(10)) + '.png'
        save_image(image_tensor, src_input, normalize = True)
        image_tensor = image_tensor.unsqueeze_(0)
        model = UNet()
        model.eval()
        model.load_state_dict(torch.load('./models/unet_bce.pth', map_location=torch.device('cpu')))
        output = model(image_tensor)
        pred = torch.sigmoid(output)
        pred = (pred > 0.5).float()
        src_output = 'static/images/'+'output'+''.join(random.choice(letters) for i in range(10)) + '.png'
        save_image(pred, src_output, normalize = True)
        return render_template('mole_segmentation.html', src_input = src_input, src_output = src_output)

    return render_template('mole.html', message = message)
    

if __name__ == '__main__':
	app.run(debug = True)