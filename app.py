from pathlib import Path
from flask import Flask, redirect, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd
import os
import urllib.request
import gdown
# ========== Google Drive Auto Model Download ==========

model_path = "plant_disease_model_1_latest.pt"

if not os.path.exists(model_path):
    print("Downloading model from Google Drive...")
    # Correct ID from your share link
    gdown.download("https://drive.google.com/uc?id=1XBo4fdRs3mihkqfIYhgnfr5a79b63ZUo", model_path, quiet=False)

# =======================================================

# Set base directory
BASE_DIR = Path(__file__).resolve().parent

# Load disease and supplement information
disease_info = pd.read_csv(BASE_DIR / "disease_info.csv", encoding='cp1252')
supplement_info = pd.read_csv(BASE_DIR / "supplement_info.csv", encoding='cp1252')

# Load CNN model
model = CNN.CNN(39)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Function to make predictions
def prediction(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.unsqueeze(0)
    with torch.no_grad():
        output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename

        upload_folder = BASE_DIR / "static"
        upload_folder.mkdir(exist_ok=True)
        file_path = upload_folder / filename
        image.save(file_path)

        pred = prediction(file_path)
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]

        return render_template('submit.html', title=title, desc=description, prevent=prevent,
                               image_url=image_url, pred=pred, sname=supplement_name,
                               simage=supplement_image_url, buy_link=supplement_buy_link)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image=list(supplement_info['supplement image']),
                           supplement_name=list(supplement_info['supplement name']),
                           disease=list(disease_info['disease_name']),
                           buy=list(supplement_info['buy link']))

# Handling 404 error

@app.errorhandler(404)
def handle_404(e):
    if request.path.startswith('/hybridaction/'):
        return '', 204  # Cleanly ignore bot/extension hits
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
