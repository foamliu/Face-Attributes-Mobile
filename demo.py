import json
import os
import pickle
import random

import cv2 as cv
import torch
from torchvision import transforms

from config import device, im_size, pickle_file_aligned, train_ratio, IMG_DIR
from data_gen import data_transforms
from utils import idx2name


def save_images(full_path, filename, i):
    raw = cv.imread(full_path)
    resized = cv.resize(raw, (im_size, im_size))
    filename = 'images/{}_raw.jpg'.format(i)
    cv.imwrite(filename, resized)

    filename = os.path.join(IMG_DIR, filename)
    img = cv.imread(filename)
    img = cv.resize(img, (im_size, im_size))
    filename = 'images/{}_img.jpg'.format(i)
    cv.imwrite(filename, img)


if __name__ == "__main__":
    with open(pickle_file_aligned, 'rb') as file:
        data = pickle.load(file)

    samples = data['samples']

    num_samples = len(samples)
    num_train = int(train_ratio * num_samples)
    samples = samples[num_train:]
    samples = random.sample(samples, 10)

    inputs = torch.zeros([10, 3, im_size, im_size], dtype=torch.float, device=device)

    transformer = data_transforms['valid']

    sample_preds = []

    for i, sample in enumerate(samples):
        filename = sample['filename']
        print(filename)
        save_images(full_path, filename, i)

        full_path = os.path.join(IMG_DIR, filename)
        # full_path = sample['filename']
        # bbox = sample['bboxes'][0]
        img = cv.imread(full_path)
        # img = crop_image(img, bbox)
        img = cv.resize(img, (im_size, im_size))

        img = cv.resize(img, (im_size, im_size))
        img = img[..., ::-1]  # RGB
        img = transforms.ToPILImage()(img)
        img = transformer(img)
        inputs[i] = img

        age = sample['attr']['age']
        pitch = sample['attr']['angle']['pitch']
        roll = sample['attr']['angle']['roll']
        yaw = sample['attr']['angle']['yaw']
        beauty = sample['attr']['beauty']
        expression = sample['attr']['expression']['type']
        face_prob = sample['attr']['face_probability']
        face_shape = sample['attr']['face_shape']['type']
        face_type = sample['attr']['face_type']['type']
        gender = sample['attr']['gender']['type']
        glasses = sample['attr']['glasses']['type']
        race = sample['attr']['race']['type']
        sample_preds.append({'i': i, 'age_true': age,
                             'pitch_true': pitch,
                             'roll_true': roll,
                             'yaw_true': yaw,
                             'beauty_true': beauty,
                             'expression_true': expression,
                             'face_prob_true': face_prob,
                             'face_shape_true': face_shape,
                             'face_type_true': face_type,
                             'gender_true': gender,
                             'glasses_true': glasses,
                             'race_true': race})

    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        reg_out, expression_out, gender_out, glasses_out, race_out = model(inputs)

    print(reg_out.size())
    reg_out = reg_out.cpu().numpy()
    age_out = reg_out[:, 0]
    pitch_out = reg_out[:, 1]
    roll_out = reg_out[:, 2]
    yaw_out = reg_out[:, 3]
    beauty_out = reg_out[:, 4]

    _, expression_out = expression_out.topk(1, 1, True, True)
    print('expression_out.size(): ' + str(expression_out.size()))
    _, gender_out = gender_out.topk(1, 1, True, True)
    print('gender_out.size(): ' + str(gender_out.size()))
    _, glasses_out = glasses_out.topk(1, 1, True, True)
    print('glasses_out.size(): ' + str(glasses_out.size()))
    _, race_out = race_out.topk(1, 1, True, True)
    print('race_out.size(): ' + str(race_out.size()))

    expression_out = expression_out.cpu().numpy()
    print('expression_out.shape: ' + str(expression_out.shape))
    gender_out = gender_out.cpu().numpy()
    print('gender_out.shape: ' + str(gender_out.shape))
    glasses_out = glasses_out.cpu().numpy()
    print('glasses_out.shape: ' + str(glasses_out.shape))
    race_out = race_out.cpu().numpy()
    print('race_out.shape: ' + str(race_out.shape))

    for i in range(10):
        sample = sample_preds[i]

        sample['age_out'] = int(age_out[i] * 100)
        sample['pitch_out'] = float('{0:.2f}'.format(pitch_out[i] * 360 - 180))
        sample['roll_out'] = float('{0:.2f}'.format(roll_out[i] * 360 - 180))
        sample['yaw_out'] = float('{0:.2f}'.format(yaw_out[i] * 360 - 180))
        sample['beauty_out'] = float('{0:.2f}'.format(beauty_out[i] * 100))
        sample['expression_out'] = idx2name(int(expression_out[i][0]), 'expression')
        sample['gender_out'] = idx2name(int(gender_out[i][0]), 'gender')
        sample['glasses_out'] = idx2name(int(glasses_out[i][0]), 'glasses')
        sample['race_out'] = idx2name(int(race_out[i][0]), 'race')

    with open('sample_preds.json', 'w') as file:
        json.dump(sample_preds, file, indent=4, ensure_ascii=False)
