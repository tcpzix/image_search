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

def train_image(image_path):
    try:
        img = Image.open(image_path)
        feature = fe.extract(img)
        feature_path = 'static/feature/' + \
            os.path.basename(image_path) + '.pkl'
        pickle.dump(feature, open(feature_path, 'wb'))
    except PIL.UnidentifiedImageError:
        print('Cannot identify image file')


if __name__ == '__main__':
    for ext in extensions:
        file_path = PATH + ext
        for image_path in glob.glob(file_path):
            image_exists = os.path.isfile('static/feature/' + \
                    os.path.basename(image_path) + '.pkl')
            if not image_exists:
                train_image(image_path)
