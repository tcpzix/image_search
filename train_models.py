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

for img_path in sorted(glob.glob(PATH + '*.jpg')):
    if os.path.isfile('static/feature/' + os.path.splitext(os.path.basename(img_path))[0] + '.pkl'):
        print(img_path, "was trained befor...")
    else:
        print("NEW IMAGE TRAINED:", img_path)
        try:
            img = Image.open(img_path)
            feature = fe.extract(img)
            feature_path = 'static/feature/' + os.path.splitext(os.path.basename(img_path))[0] + '.pkl'
            pickle.dump(feature, open(feature_path, 'wb'))
        except PIL.UnidentifiedImageError:
            print('Cannot identify image file')
