import pickle

import cv2 as cv
from tqdm import tqdm

from config import im_size, pickle_file_landmarks, pickle_file_aligned
from utils import crop_image

ONE_SECOND = 1

if __name__ == "__main__":
    print('loading {}...'.format(pickle_file_landmarks))
    with open(pickle_file_landmarks, 'rb') as file:
        data = pickle.load(file)

    items = data['samples']
    print('num_items: ' + str(len(items)))

    samples = []
    for item in tqdm(items):
        try:
            full_path = item['full_path']
            bbox = item['bboxes'][0]
            img = cv.imread(full_path)
            img = crop_image(img, bbox)
            img = cv.resize(img, (im_size, im_size))
            samples.append(item)
        except:
            pass

    print('num_items: ' + str(len(samples)))

    print('saving {}...'.format(pickle_file_aligned))
    with open(pickle_file_aligned, 'wb') as file:
        save = {
            'samples': samples
        }
        pickle.dump(save, file, pickle.HIGHEST_PROTOCOL)
