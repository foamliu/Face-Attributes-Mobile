# Face-Attributes

Deep Face Attributes.


## DataSet

CASIA WebFace DataSet, 479,653 faces.

### Gender

![image](https://github.com/foamliu/Face-Attributes/raw/master/images/gender_dist.png)

### Age

![image](https://github.com/foamliu/Face-Attributes/raw/master/images/age_dist.png)

### Euler angles:

![image](https://github.com/foamliu/Face-Attributes/raw/master/images/euler_angles.png)

Pitch:

![image](https://github.com/foamliu/Face-Attributes/raw/master/images/angle_pitch_dist.png)

Yaw:

![image](https://github.com/foamliu/Face-Attributes/raw/master/images/angle_yaw_dist.png)

Roll:

![image](https://github.com/foamliu/Face-Attributes/raw/master/images/angle_roll_dist.png)

### Beauty

![image](https://github.com/foamliu/Face-Attributes/raw/master/images/beauty_dist.png)

### Expression

![image](https://github.com/foamliu/Face-Attributes/raw/master/images/expression_dist.png)

### Face shape

![image](https://github.com/foamliu/Face-Attributes/raw/master/images/face_shape_dist.png)

### Face type

![image](https://github.com/foamliu/Face-Attributes/raw/master/images/face_type_dist.png)

### Glasses

![image](https://github.com/foamliu/Face-Attributes/raw/master/images/glasses_dist.png)

### Race

![image](https://github.com/foamliu/Face-Attributes/raw/master/images/race_dist.png)

## Dependencies
- PyTorch 1.0.0

## Usage


### Train
```bash
$ python train.py
```

To visualize the training process：
```bash
$ tensorboard --logdir=runs
```

![image](https://github.com/foamliu/Face-Attributes/raw/master/images/learning_curve.jpg)

Image | Aligned | Out | True |
|---|---|---|---|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/0_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/0_img.jpg)|age: 31<br>pitch: 7.43<br>roll: 4.03<br>yaw: -4.39<br>beauty: 37.45<br>expression: smile<br>gender: male<br>glasses: none<br>race: white|age: 30<br>pitch: 9.56<br>roll: 3.02<br>yaw: -8.24<br>beauty: 37.73<br>expression: smile<br>gender: male<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/1_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/1_img.jpg)|age: 23<br>pitch: 7.24<br>roll: 1.08<br>yaw: -36.76<br>beauty: 51.83<br>expression: none<br>gender: female<br>glasses: none<br>race: white|age: 23<br>pitch: 6.15<br>roll: 1.63<br>yaw: -33.32<br>beauty: 57.46<br>expression: none<br>gender: female<br>glasses: common<br>race: yellow|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/2_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/2_img.jpg)|age: 35<br>pitch: 17.6<br>roll: -9.36<br>yaw: 36.16<br>beauty: 39.17<br>expression: none<br>gender: male<br>glasses: none<br>race: white|age: 36<br>pitch: 12.3<br>roll: -10.03<br>yaw: 33.63<br>beauty: 38.63<br>expression: none<br>gender: male<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/3_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/3_img.jpg)|age: 26<br>pitch: 11.08<br>roll: -10.73<br>yaw: -55.27<br>beauty: 42.05<br>expression: none<br>gender: female<br>glasses: none<br>race: white|age: 26<br>pitch: 10.39<br>roll: -13.08<br>yaw: -49.21<br>beauty: 49.31<br>expression: none<br>gender: female<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/4_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/4_img.jpg)|age: 30<br>pitch: 3.83<br>roll: -8.48<br>yaw: 2.38<br>beauty: 46.39<br>expression: smile<br>gender: male<br>glasses: none<br>race: white|age: 31<br>pitch: 6.5<br>roll: -9.29<br>yaw: 4.35<br>beauty: 35.32<br>expression: none<br>gender: male<br>glasses: sun<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/5_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/5_img.jpg)|age: 36<br>pitch: 11.59<br>roll: -7.07<br>yaw: 0.48<br>beauty: 34.55<br>expression: smile<br>gender: female<br>glasses: none<br>race: white|age: 35<br>pitch: 11.67<br>roll: -6.98<br>yaw: -1.25<br>beauty: 35.31<br>expression: smile<br>gender: female<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/6_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/6_img.jpg)|age: 20<br>pitch: -3.13<br>roll: 19.29<br>yaw: 51.62<br>beauty: 53.43<br>expression: none<br>gender: male<br>glasses: none<br>race: white|age: 23<br>pitch: -4.04<br>roll: 17.68<br>yaw: 54.1<br>beauty: 48.18<br>expression: none<br>gender: male<br>glasses: none<br>race: yellow|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/7_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/7_img.jpg)|age: 33<br>pitch: 7.76<br>roll: -0.08<br>yaw: -1.68<br>beauty: 55.65<br>expression: none<br>gender: female<br>glasses: none<br>race: white|age: 31<br>pitch: 7.35<br>roll: 1.28<br>yaw: -5.05<br>beauty: 55.49<br>expression: none<br>gender: female<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/8_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/8_img.jpg)|age: 37<br>pitch: 15.85<br>roll: -1.86<br>yaw: -10.53<br>beauty: 41.24<br>expression: none<br>gender: male<br>glasses: none<br>race: white|age: 41<br>pitch: 17.6<br>roll: -2.29<br>yaw: -14.45<br>beauty: 40.19<br>expression: none<br>gender: male<br>glasses: none<br>race: black|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/9_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/9_img.jpg)|age: 39<br>pitch: 10.82<br>roll: -8.91<br>yaw: -5.19<br>beauty: 40.53<br>expression: smile<br>gender: female<br>glasses: none<br>race: white|age: 41<br>pitch: 11.72<br>roll: -9.31<br>yaw: -4.21<br>beauty: 35.14<br>expression: smile<br>gender: female<br>glasses: none<br>race: white|
