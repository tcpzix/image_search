import glob
import os
import pickle
import PIL
from PIL import Image
from feature_extractor import FeatureExtractor
from dotenv import load_dotenv


# load .env variables
load_dotenv()

PATH = os.environ.get('IMAGES_PATH')

fe = FeatureExtractor()

extensions = ['*.1.jpg', '*.1.jpeg', '*.1.png']
all_files = []

for ext in extensions:
    file_path = PATH + ext
    for img_path in glob.glob(file_path):
        if os.path.isfile('static/feature/' + \
                os.path.basename(img_path) + '.pkl'):
            print(img_path, "was trained befor...")
        else:
            print("Traning new image: {}".format(img_path))
            try:
                img = Image.open(img_path)
                feature = fe.extract(img)
                feature_path = 'static/feature/' + \
                    os.path.basename(img_path) + '.pkl'
                pickle.dump(feature, open(feature_path, 'wb'))
            except PIL.UnidentifiedImageError:
                print('Cannot identify image file')
