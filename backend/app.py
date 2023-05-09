from flask import Flask, request, jsonify
import numpy as np
import cv2
import torch
import joblib
import pandas as pd
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from skimage.io import imread, imshow, imsave
from skimage import data
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
import cv2
from skimage.measure.entropy import shannon_entropy
from PIL import Image
from skimage import io
from scipy.stats import skew

app = Flask(__name__)
model_yolo = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp2/weights/best.pt')  # local model
model_apple1 = joblib.load('models/applemodel_1.joblib')
model_apple2 = joblib.load('models/applemodel_2.joblib')
model_banana = joblib.load('models/bananamodel.joblib')
model_tomato = joblib.load('models/tomatomodel.joblib')
preprocess_apple1 = joblib.load('models/AppleScaler_1.save')
preprocess_apple2 = joblib.load('models/AppleScaler_2.save')
preprocess_banana2 = joblib.load('models/BananaScaler.save')

def entropy_mask_viz(image,factor):
    image_gray = rgb2gray(image)
    entropy_image = entropy(image_gray, disk(4),)
    scaled_entropy = entropy_image / entropy_image.max()
    f_size = 24
    threshold = scaled_entropy > factor
    image_a = np.dstack([image[:,:,0]*threshold,
                            image[:,:,1]*threshold,
                            image[:,:,2]*threshold])
    return image_a

def test(image,fruit):
    if fruit=='apple':
        factor = 0.3
    elif fruit=='banana':
        factor = 0.25
    else:
        factor = 0.25
    img = entropy_mask_viz(image,factor)
    img1 = img[..., :3]
    x = np.array(img)
    # angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    img = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
#             Statistical Features
    R, G, B = np.dsplit(img1, img1.shape[-1])
    mean_R=(np.mean(R))
    mean_G=(np.mean(G))
    mean_B=(np.mean(B))
    std_R=(np.std(R))
    std_G=(np.std(G))
    std_B=(np.std(B))
    skewness_R=(skew(R.flatten()))
    skewness_G=(skew(G.flatten()))
    skewness_B=(skew(B.flatten()))
#             Texture Feature
    glcm_0 = greycomatrix(img, [5], [0], levels=256, normed=True, symmetric=True)
    glcm_45 = greycomatrix(img, [5], [np.pi/4], levels=256, normed=True, symmetric=True)
    glcm_90 = greycomatrix(img, [5], [np.pi/2], levels=256, normed=True, symmetric=True)
    glcm_135 = greycomatrix(img, [5], [3*np.pi/4], levels=256, normed=True, symmetric=True)
#             Contrast
    contrast_0=(greycoprops(glcm_0, 'contrast')[0][0])
    contrast_45=(greycoprops(glcm_45, 'contrast')[0][0])
    contrast_90=(greycoprops(glcm_90, 'contrast')[0][0])
    contrast_135=(greycoprops(glcm_135, 'contrast')[0][0])
#             Correlation
    correlation_0=(greycoprops(glcm_0, 'correlation')[0][0])
    correlation_45=(greycoprops(glcm_45, 'correlation')[0][0])
    correlation_90=(greycoprops(glcm_90, 'correlation')[0][0])
    correlation_135=(greycoprops(glcm_135, 'correlation')[0][0])
#             Energy
    energy_0=(greycoprops(glcm_0, 'energy')[0][0])
    energy_45=(greycoprops(glcm_45, 'energy')[0][0])
    energy_90=(greycoprops(glcm_90, 'energy')[0][0])
    energy_135=(greycoprops(glcm_135, 'energy')[0][0])
#             Homogeneity
    homogeneity_0=(greycoprops(glcm_0, 'homogeneity')[0][0])
    homogeneity_45=(greycoprops(glcm_45, 'homogeneity')[0][0])
    homogeneity_90=(greycoprops(glcm_90, 'homogeneity')[0][0])
    homogeneity_135=(greycoprops(glcm_135, 'homogeneity')[0][0])
    d = {'Mean_R':mean_R,'Mean_G':mean_G,'Mean_B':mean_B,'Std_R':std_R,'Std_G':std_G,
         'Std_B':std_B,'Skew_R':skewness_R,'Skew_G':skewness_G,'Skew_B':skewness_B,
         'Contrast_0':contrast_0,'Correlation_0':correlation_0,'Energy_0':energy_0,'Homogeneity_0':homogeneity_0,
        'Contrast_45':contrast_45,'Correlation_45':correlation_45,'Energy_45':energy_45,'Homogeneity_45':homogeneity_45,
        'Contrast_90':contrast_90,'Correlation_90':correlation_90,'Energy_90':energy_90,'Homogeneity_90':homogeneity_90,
        'Contrast_135':contrast_135,'Correlation_135':correlation_135,'Energy_135':energy_135,'Homogeneity_135':homogeneity_135}
    return img, d

@app.route('/')
def welcome_check():
    return "Welcome to this page which means the app is working"

@app.route('/predict_type', methods=['POST'])
def predict_type():    
    im = request.files['image']
    print(f"Image: {im}")
    im = Image.open(im)
    results = model_yolo(im)
    class_index = int(results.pred[0][0][-1])
    class_label = results.names[class_index]
    # print(class_label)
    return class_label

@app.route('/predict_fresh', methods=['POST'])
def predict_fresh():    
    image = request.form['image']
    img = imread(image)
    label = request.form['label']
    mask, features = test(img,label)
    
    if label == 'apple':
        dct = {k:[v] for k,v in features.items()}
        df = pd.DataFrame(dct)
        X1 = preprocess_apple1.transform(df)
        X2 = preprocess_apple2.transform(df)
        pred1 = model_apple1.predict(X1)
        if pred1[0] ==0:
            return 'Fresh - 0% Rotten'
        else:
            pred2 = model_apple2.predict(X2)
            if pred2[0] == 1:
                return "35% Rotten"
            elif pred2[0]==2:
                return '70% Rotten'
            else:
                return '100% Rotten'
    elif label == 'banana':
        mask, features = test(img,label)
        dct = {k:[v] for k,v in features.items()}  
        df = pd.DataFrame(dct)
        X = preprocess_apple1.transform(df)
        pred = model_banana.predict(X)
        if pred[0]==0:
            return '0% Rotten'
        elif pred[0]==1:
            return '35% Rotten'
        elif pred[0]==2:
            return '70% Rotten'
        else:
            return '100% Rotten'

    elif label == 'tomato':
        mask, features = test(img,label)
        dct = {k:[v] for k,v in features.items()} 
        df = pd.DataFrame(dct)
        # X = preprocess_apple1.transform(df)
        pred = model_tomato.predict(df)
        if pred[0]==0:
            return "0% Rotten"
        elif pred[0]==1:
            return "50% Rotten"
        else:
            return "100% Rotten"
        
    
# if __name__ == '__main__':
#     app.run()