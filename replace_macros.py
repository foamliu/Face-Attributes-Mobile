# -*- coding: utf-8 -*-
import json


def safe_get(dic, key):
    if key in dic:
        return dic[key]
    else:
        return None


def get_attrs(item, split):
    age = safe_get(item, 'age_' + split)
    pitch = safe_get(item, 'pitch_' + split)
    roll = safe_get(item, 'roll_' + split)
    yaw = safe_get(item, 'yaw_' + split)
    beauty = safe_get(item, 'beauty_' + split)
    expression = safe_get(item, 'expression_' + split)
    # face_prob = safe_get(item, 'face_prob_' + split)
    # face_shape = safe_get(item, 'face_shape_' + split)
    # face_type = safe_get(item, 'face_type_' + split)
    gender = safe_get(item, 'gender_' + split)
    glasses = safe_get(item, 'glasses_' + split)
    race = safe_get(item, 'race_' + split)
    result = 'age: {}<br>'.format(age)
    result += 'pitch: {}<br>'.format(pitch)
    result += 'roll: {}<br>'.format(roll)
    result += 'yaw: {}<br>'.format(yaw)
    result += 'beauty: {}<br>'.format(beauty)
    result += 'expression: {}<br>'.format(expression)
    # result += 'face_prob: {}<br>'.format(face_prob)
    # result += 'face_shape: {}<br>'.format(face_shape)
    # result += 'face_type: {}<br>'.format(face_type)
    result += 'gender: {}<br>'.format(gender)
    result += 'glasses: {}<br>'.format(glasses)
    result += 'race: {}'.format(race)
    return result


if __name__ == '__main__':
    with open('README.t', 'r', encoding="utf-8") as file:
        text = file.readlines()
    text = ''.join(text)

    with open('sample_preds.json', 'r', encoding="utf-8") as file:
        results = json.load(file)

    for i in range(10):
        item = results[i]
        result_true = get_attrs(item, 'true')
        result_out = get_attrs(item, 'out')
        text = text.replace('$(result_true_{})'.format(i), result_true)
        text = text.replace('$(result_out_{})'.format(i), result_out)

    with open('README.md', 'w', encoding="utf-8") as file:
        file.write(text)
