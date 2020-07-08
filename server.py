import os
import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
import glob
import pickle
from datetime import datetime
from flask import Flask, request, render_template

app = Flask(__name__) 

fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in glob.glob("static/feature/*"):
    features.append(pickle.load(open(feature_path, 'rb')))
    img_paths.append('static/img/' + os.path.splitext(os.path.basename(feature_path))[0] + '.jpg')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        img = Image.open(file.stream)
        uploaded_img_path = "static/uploaded/" + datetime.now().strftime('%Y_%m_%d_%H_%M_%S')  + "_" + file.filename
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
    

    
# api for do search and send back result


@app.route('/imagesearch', methods=['POST'])
def directSearch():
    req = request.get_json()
    file_path = req["image_address"]
    result_count = req["result_count"]
    # get file name
    dirname = os.path.dirname(__file__)
    file = os.path.join(dirname, file_path)
    filename = os.path.basename(file_path)
    # open file
    img = Image.open(file)
    # do search
    query = fe.extract(img)
    dists = np.linalg.norm(features - query, axis=1)
    ids = np.argsort(dists)[:result_count]
    scores = [(img_paths[id], str(dists[id])) for id in ids]
    # result
    data = dict(scores)
    return data



# send file path in url like: http://x.x.x.x/imgsearch/c:/cat.jpg
@app.route('/imgsearch/<path:path>', methods=['GET', 'POST'])
def search(path):
    # convert file path str to raw
    file = r"{}".format(path)
    # get file name only
    filename = os.path.basename(file)
    # open file
    img = Image.open(file)
    # do search 
    query = fe.extract(img)
    dists = np.linalg.norm(features - query, axis=1)
    ids = np.argsort(dists)[:10] # number of result
    scores = [(img_paths[id], str(dists[id])) for id in ids]
    # result
    data = dict(scores)
    return data


if __name__=="__main__": 
    app.run("0.0.0.0")
 
