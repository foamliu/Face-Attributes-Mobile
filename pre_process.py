import os
import pickle

import cv2 as cv
from tqdm import tqdm

from config import pickle_file, IMG_DIR
from retinaface.detector import detect_faces
from utils import ensure_folder


def select_significant_face(bboxes):
    best_index = -1
    best_rank = float('-inf')
    for i, b in enumerate(bboxes):
        bbox_w, bbox_h = b[2] - b[0], b[3] - b[1]
        area = bbox_w * bbox_h
        score = b[4]
        rank = score * area
        if rank > best_rank:
            best_rank = rank
            best_index = i

    return best_index


if __name__ == "__main__":
    ensure_folder(IMG_DIR)

    print('loading {}...'.format(pickle_file))
    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)

    items = data['samples']
    print('num_items: ' + str(len(items)))

    samples = []
    for item in tqdm(items):
        try:
            full_path = item['full_path']
            img = cv.imread(full_path)
            bboxes, landmarks = detect_faces(img)
            idx = select_significant_face(bboxes)
            b = bboxes[idx]
            img = img[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
            filename = full_path.replace('data/CASIA-WebFace/', '').replace('/', '_')
            filename = os.path.join(IMG_DIR, filename)
            cv.imwrite(filename, img)
        except Exception as err:
            print(err)
