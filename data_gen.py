import pickle

import cv2 as cv
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from config import im_size, pickle_file_aligned, num_train
from utils import crop_image, name2idx

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class FaceAttributesDataset(Dataset):
    def __init__(self, split):
        with open(pickle_file_aligned, 'rb') as file:
            data = pickle.load(file)

        samples = data['samples']

        if split == 'train':
            self.samples = samples[:num_train]
        else:
            self.samples = samples[num_train:]

        self.transformer = data_transforms[split]

    def __getitem__(self, i):
        sample = self.samples[i]
        full_path = sample['full_path']
        bbox = sample['bboxes'][0]
        img = cv.imread(full_path)
        img = crop_image(img, bbox)
        img = cv.resize(img, (im_size, im_size))

        # img aug
        img = img[..., ::-1]  # RGB
        img = transforms.ToPILImage()(img)
        img = self.transformer(img)

        age = sample['attr']['age'] / 100.
        pitch = (sample['attr']['angle']['pitch'] + 180) / 360
        roll = (sample['attr']['angle']['roll'] + 180) / 360
        yaw = (sample['attr']['angle']['yaw'] + 180) / 360
        beauty = sample['attr']['beauty'] / 100.

        expression = name2idx(sample['attr']['expression']['type'])
        gender = name2idx(sample['attr']['gender']['type'])
        glasses = name2idx(sample['attr']['glasses']['type'])
        race = name2idx(sample['attr']['race']['type'])
        return img, np.array([age, pitch, roll, yaw, beauty]), expression, gender, glasses, race

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    dataset = FaceAttributesDataset('train')
    print(dataset[0])
