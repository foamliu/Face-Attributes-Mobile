import pickle

from tqdm import tqdm

from config import pickle_file, pickle_file_landmarks
from utils import get_face_attributes

ONE_SECOND = 1

if __name__ == "__main__":
    print('loading {}...'.format(pickle_file))
    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)

    items = data['samples']
    print('num_items: ' + str(len(items)))

    samples = []
    for item in tqdm(items):
        filename = item['full_path']
        is_valid, bboxes, landmarks = get_face_attributes(filename)
        if is_valid:
            sample = item
            sample['bboxes'] = bboxes
            sample['landmarks'] = landmarks
            samples.append(sample)

    print('saving {}...'.format(pickle_file_landmarks))
    with open(pickle_file_landmarks, 'wb') as file:
        save = {
            'samples': samples
        }
        pickle.dump(save, file, pickle.HIGHEST_PROTOCOL)
