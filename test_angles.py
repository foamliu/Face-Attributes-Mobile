import pickle
import random

import cv2 as cv

from config import im_size, pickle_file_landmarks, train_ratio
from utils import crop_image
from utils_aip import get_face_attributes


def save_images(full_path, i, bbox):
    img = cv.imread(full_path)
    img = crop_image(img, bbox)
    img = cv.resize(img, (im_size, im_size))
    img_path1 = 'images/img_norm_{}.jpg'.format(i)
    cv.imwrite(img_path1, img)

    img = cv.flip(img, 1)
    img_path2 = 'images/img_flip_{}.jpg'.format(i)
    cv.imwrite(img_path2, img)
    return img_path1, img_path2


if __name__ == "__main__":
    with open(pickle_file_landmarks, 'rb') as file:
        data = pickle.load(file)

    samples = data['samples']

    num_samples = len(samples)
    num_train = int(train_ratio * num_samples)
    samples = samples[num_train:]
    samples = random.sample(samples, 10)

    for i, sample in enumerate(samples):
        full_path = sample['full_path']
        bbox = sample['bboxes'][0]
        print(full_path)
        img_path1, img_path2 = save_images(full_path, i, bbox)

        attr = get_face_attributes(img_path1)
        if attr:
            print(attr['angle'])

        attr = get_face_attributes(img_path2)
        if attr:
            print(attr['angle'])
