import os
import numpy as np
import glob
import pickle
from datetime import datetime
from PIL import Image
from feature_extractor import FeatureExtractor
from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv


# load .env variables
load_dotenv()

PATH = os.environ.get('IMAGES_PATH')


app = Flask(__name__) 


fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in glob.glob("static/feature/*"):
    features.append(pickle.load(open(feature_path, 'rb')))
    img_paths.append(PATH + \
        os.path.splitext(os.path.basename(feature_path))[0] + '.jpg')



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        img = Image.open(file.stream)
        uploaded_img_path = "static/uploaded/" + \
            datetime.now().strftime('%Y_%m_%d_%H_%M_%S')  + "_" + file.filename
        img.save(uploaded_img_path)
       
        query = fe.extract(img)
        dists = np.linalg.norm(features - query, axis=1)
        ids = np.argsort(dists)[:10] 
        scores = [(dists[id], img_paths[id]) for id in ids]

        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')



@app.route('/imagesearch/', methods=['POST'])
def search():
    """API endpoint for searching and sending back most similar images"""
    json_data = request.get_json()
    image_address = json_data.get("image_address", None)
    limit = json_data.get("limit", None)

    if image_address and limit:
        # get file name
        dirname = os.path.dirname(__file__)
        file = os.path.join(dirname, image_address)
        
        # open file
        img = Image.open(file)
        
        # do search
        query = fe.extract(img)
        dists = np.linalg.norm(features - query, axis=1)
        ids = np.argsort(dists)[:limit]
        scores = [(img_paths[id], str(dists[id])) for id in ids]

        result = []
        for image_path, score in scores:
            result.append({
                'image_path': image_path,
                'score': score
            })

        # result
        # data = dict(scores)
        return jsonify(result)

    # error, fields not set
    return {
        'detail': 'image_address and limit fields are required'
    }, 400


if __name__=="__main__": 
    app.run()
