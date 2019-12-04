import argparse
import logging

import cv2 as cv
import numpy as np
import torch
from PIL import Image

from align_faces import get_reference_facial_points, warp_and_crop_face
from config import image_h, image_w
from retinaface.detector import detect_faces


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(epoch, epochs_since_improvement, model, optimizer, loss, is_best):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'loss': loss,
             'model': model,
             'optimizer': optimizer}
    # filename = 'checkpoint_' + str(epoch) + '_' + str(loss) + '.tar'
    filename = 'checkpoint.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_checkpoint.tar')


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LossMeterBag(object):

    def __init__(self, name_list):
        self.meter_dict = dict()
        self.name_list = name_list
        for name in self.name_list:
            self.meter_dict[name] = AverageMeter()

    def update(self, val_list):
        for i, name in enumerate(self.name_list):
            val = val_list[i]
            self.meter_dict[name].update(val)

    def __str__(self):
        ret = ''
        for name in self.name_list:
            ret += '{0}:\t {1:.4f}({2:.4f})\t'.format(name, self.meter_dict[name].val, self.meter_dict[name].avg)

        return ret


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k=1):
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')
    # general
    parser.add_argument('--end-epoch', type=int, default=50, help='training epoch size.')
    parser.add_argument('--lr', type=float, default=0.001, help='start learning rate')
    parser.add_argument('--lr-step', type=int, default=10, help='period of learning rate decay')
    parser.add_argument('--optimizer', default='sgd', help='optimizer')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--batch-size', type=int, default=512, help='batch size in each context')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint')
    parser.add_argument('--use-se', type=bool, default=False, help='use SEBlock')
    parser.add_argument('--pretrained', type=bool, default=False, help='pretrained model')
    args = parser.parse_args()
    return args


def get_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s \t%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


def get_face_attributes(full_path):
    try:
        img = Image.open(full_path).convert('RGB')
        bounding_boxes, landmarks = detect_faces(img)

        if len(landmarks) > 0:
            landmarks = [int(round(x)) for x in landmarks[0]]
            return True, bounding_boxes, landmarks

    except KeyboardInterrupt:
        raise
    except:
        pass
    return False, None, None


def align_face(img_fn, facial5points):
    raw = cv.imread(img_fn, True)
    facial5points = np.reshape(facial5points, (2, 5))

    crop_size = (image_h, image_w)

    default_square = True
    inner_padding_factor = 0.25
    outer_padding = (0, 0)
    output_size = (image_h, image_w)

    # get the reference 5 landmarks position in the crop settings
    reference_5pts = get_reference_facial_points(
        output_size, inner_padding_factor, outer_padding, default_square)

    # dst_img = warp_and_crop_face(raw, facial5points)
    dst_img = warp_and_crop_face(raw, facial5points, reference_pts=reference_5pts, crop_size=crop_size)
    return dst_img


expression_dict = {0: 'none', 1: 'smile', 2: 'laugh'}
face_shape_dict = {0: 'square', 1: 'oval', 2: 'heart', 3: 'round', 4: 'triangle'}
face_type_dict = {0: 'human', 1: 'cartoon'}
gender_dict = {0: 'female', 1: 'male'}
glasses_dict = {0: 'none', 1: 'sun', 2: 'common'}
race_dict = {0: 'yellow', 1: 'white', 2: 'black', 3: 'arabs'}


def idx2name(idx, tag):
    name = None
    if tag == 'expression':
        name = expression_dict[idx]
    elif tag == 'face_shape':
        name = face_shape_dict[idx]
    elif tag == 'face_type':
        name = face_type_dict[idx]
    elif tag == 'gender':
        name = gender_dict[idx]
    elif tag == 'glasses':
        name = glasses_dict[idx]
    elif tag == 'race':
        name = race_dict[idx]
    return name


def name2idx(name):
    lookup_table = {'none': 0, 'smile': 1, 'laugh': 2,
                    'square': 0, 'oval': 1, 'heart': 2, 'round': 3, 'triangle': 4,
                    'human': 0, 'cartoon': 1,
                    'female': 0, 'male': 1,
                    'sun': 1, 'common': 2,
                    'yellow': 0, 'white': 1, 'black': 2, 'arabs': 3}

    return lookup_table[name]


def crop_image(img, bbox):
    # print(bbox.shape)
    x1 = int(round(bbox[0]))
    y1 = int(round(bbox[1]))
    x2 = int(round(bbox[2]))
    y2 = int(round(bbox[3]))
    w = int(abs(x2 - x1))
    h = int(abs(y2 - y1))
    # print(x1, y1, w, h)
    # print(img.shape)
    # print('x1:{} y1:{} w:{} h:{}'.format(x1, y1, w, h))
    crop_img = img[y1:y1 + h, x1:x1 + w]
    return crop_img
