# Face Attributes Mobile

Regress Face Attributes with MobileNetV2.

## Features
1. Estimate Gender, Age, Euler angles, Beauty, Expression, Face shape, Face type and Glasses with a single image.
2. Lightweight: Params size (MB): 2.14, FLOPs size (GB): 0.32, Total Size (MB): 9.18.


## DataSet

CASIA WebFace DataSet, 479,653 faces.

### Gender

![image](https://github.com/foamliu/Face-Attributes-v2/raw/master/images/gender_dist.png)

### Age

![image](https://github.com/foamliu/Face-Attributes-v2/raw/master/images/age_dist.png)

### Euler angles:

![image](https://github.com/foamliu/Face-Attributes-v2/raw/master/images/euler_angles.png)

Pitch:

![image](https://github.com/foamliu/Face-Attributes-v2/raw/master/images/angle_pitch_dist.png)

Yaw:

![image](https://github.com/foamliu/Face-Attributes-v2/raw/master/images/angle_yaw_dist.png)

Roll:

![image](https://github.com/foamliu/Face-Attributes-v2/raw/master/images/angle_roll_dist.png)

### Beauty

![image](https://github.com/foamliu/Face-Attributes-v2/raw/master/images/beauty_dist.png)

### Expression

![image](https://github.com/foamliu/Face-Attributes-v2/raw/master/images/expression_dist.png)

### Face shape

![image](https://github.com/foamliu/Face-Attributes-v2/raw/master/images/face_shape_dist.png)

### Face type

![image](https://github.com/foamliu/Face-Attributes-v2/raw/master/images/face_type_dist.png)

### Glasses

![image](https://github.com/foamliu/Face-Attributes-v2/raw/master/images/glasses_dist.png)

### Race

![image](https://github.com/foamliu/Face-Attributes-v2/raw/master/images/race_dist.png)

## Dependencies
- Python 3.6.8
- PyTorch 1.3.0

## Usage


### Train
```bash
$ python train.py
```

To visualize the training process：
```bash
$ tensorboard --logdir=runs
```

![image](https://github.com/foamliu/Face-Attributes-v2/raw/master/images/learning_curve.jpg)

Image | Aligned | Out | True |
|---|---|---|---|
|![image](https://github.com/foamliu/Face-Attributes-v2/raw/master/images/0_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes-v2/raw/master/images/0_img.jpg)|age: 31<br>pitch: -5.58<br>roll: 0.83<br>yaw: -24.83<br>beauty: 47.2<br>expression: none<br>gender: female<br>glasses: none<br>race: white|age: 32<br>pitch: -4.77<br>roll: 1.15<br>yaw: -26.22<br>beauty: 57.96<br>expression: none<br>gender: female<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes-v2/raw/master/images/1_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes-v2/raw/master/images/1_img.jpg)|age: 32<br>pitch: 10.87<br>roll: -4.92<br>yaw: -20.07<br>beauty: 66.32<br>expression: none<br>gender: male<br>glasses: none<br>race: white|age: 31<br>pitch: 13.28<br>roll: -5.81<br>yaw: -18.85<br>beauty: 65.91<br>expression: none<br>gender: male<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes-v2/raw/master/images/2_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes-v2/raw/master/images/2_img.jpg)|age: 38<br>pitch: 12.82<br>roll: -0.2<br>yaw: -12.13<br>beauty: 35.49<br>expression: none<br>gender: male<br>glasses: none<br>race: white|age: 42<br>pitch: 13.01<br>roll: -0.03<br>yaw: -16.77<br>beauty: 50.19<br>expression: none<br>gender: male<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes-v2/raw/master/images/3_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes-v2/raw/master/images/3_img.jpg)|age: 22<br>pitch: 4.57<br>roll: -7.73<br>yaw: 23.17<br>beauty: 43.42<br>expression: none<br>gender: female<br>glasses: none<br>race: white|age: 23<br>pitch: 5.51<br>roll: -7.73<br>yaw: 18.73<br>beauty: 45.36<br>expression: none<br>gender: female<br>glasses: sun<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes-v2/raw/master/images/4_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes-v2/raw/master/images/4_img.jpg)|age: 34<br>pitch: 8.17<br>roll: -5.39<br>yaw: -0.97<br>beauty: 60.92<br>expression: none<br>gender: male<br>glasses: none<br>race: white|age: 37<br>pitch: 7.48<br>roll: -6.35<br>yaw: -0.35<br>beauty: 62.51<br>expression: none<br>gender: male<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes-v2/raw/master/images/5_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes-v2/raw/master/images/5_img.jpg)|age: 28<br>pitch: 9.68<br>roll: 11.88<br>yaw: -40.47<br>beauty: 49.96<br>expression: none<br>gender: female<br>glasses: none<br>race: white|age: 28<br>pitch: 7.41<br>roll: 12.21<br>yaw: -38.69<br>beauty: 48.13<br>expression: none<br>gender: female<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes-v2/raw/master/images/6_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes-v2/raw/master/images/6_img.jpg)|age: 35<br>pitch: 17.0<br>roll: -5.98<br>yaw: -3.54<br>beauty: 70.58<br>expression: none<br>gender: female<br>glasses: none<br>race: white|age: 37<br>pitch: 17.99<br>roll: -4.34<br>yaw: -7.96<br>beauty: 66.77<br>expression: none<br>gender: female<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes-v2/raw/master/images/7_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes-v2/raw/master/images/7_img.jpg)|age: 30<br>pitch: 20.61<br>roll: 12.72<br>yaw: -22.68<br>beauty: 59.12<br>expression: smile<br>gender: male<br>glasses: none<br>race: white|age: 28<br>pitch: 19.27<br>roll: 13.28<br>yaw: -26.46<br>beauty: 58.69<br>expression: smile<br>gender: male<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes-v2/raw/master/images/8_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes-v2/raw/master/images/8_img.jpg)|age: 46<br>pitch: 11.58<br>roll: -3.16<br>yaw: 19.05<br>beauty: 56.29<br>expression: none<br>gender: male<br>glasses: none<br>race: white|age: 45<br>pitch: 8.26<br>roll: -1.08<br>yaw: 20.05<br>beauty: 61.68<br>expression: none<br>gender: male<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes-v2/raw/master/images/9_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes-v2/raw/master/images/9_img.jpg)|age: 34<br>pitch: 17.02<br>roll: -1.11<br>yaw: -3.17<br>beauty: 62.15<br>expression: none<br>gender: male<br>glasses: none<br>race: white|age: 34<br>pitch: 18.41<br>roll: -0.76<br>yaw: -5.72<br>beauty: 55.57<br>expression: none<br>gender: male<br>glasses: none<br>race: white|
