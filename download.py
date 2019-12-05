import os
import pickle

from ratelimit import limits, sleep_and_retry
from tqdm import tqdm

from config import pickle_file
from utils_aip import get_face_attributes

ONE_SECOND = 1


@sleep_and_retry
@limits(calls=2, period=ONE_SECOND)
def get_attr(fn):
    return get_face_attributes(fn)


if __name__ == "__main__":
    subjects = [d for d in os.listdir('data/CASIA-WebFace') if os.path.isdir(os.path.join('data/CASIA-WebFace', d))]
    assert (len(subjects) == 10575), "Number of subjects is: {}!".format(len(subjects))

    file_names = []
    for i in range(len(subjects)):
        sub = subjects[i]
        folder = os.path.join('data/CASIA-WebFace', sub)
        files = [f for f in os.listdir(folder) if
                 os.path.isfile(os.path.join(folder, f)) and f.lower().endswith('.jpg')]
        for file in files:
            filename = os.path.join(folder, file)
            file_names.append({'filename': filename, 'class_id': i, 'subject': sub})

    assert (len(file_names) == 494414), "Number of files is: {}!".format(len(file_names))

    samples = []
    for item in tqdm(file_names):
        filename = item['filename']
        try:
            attr = get_attr(filename)

            if len(attr) > 0:
                class_id = item['class_id']
                sub = item['subject']

                samples.append(
                    {'class_id': class_id, 'subject': sub, 'full_path': filename, 'attr': attr})
        except Exception as err:
            print(err)

    print('num_samples: ' + str(len(samples)))

    with open(pickle_file, 'wb') as file:
        save = {
            'samples': samples
        }
        pickle.dump(save, file, pickle.HIGHEST_PROTOCOL)
