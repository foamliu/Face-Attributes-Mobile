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
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/0_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/0_img.jpg)|$(result_out_0)|$(result_true_0)|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/1_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/1_img.jpg)|$(result_out_1)|$(result_true_1)|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/2_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/2_img.jpg)|$(result_out_2)|$(result_true_2)|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/3_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/3_img.jpg)|$(result_out_3)|$(result_true_3)|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/4_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/4_img.jpg)|$(result_out_4)|$(result_true_4)|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/5_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/5_img.jpg)|$(result_out_5)|$(result_true_5)|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/6_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/6_img.jpg)|$(result_out_6)|$(result_true_6)|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/7_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/7_img.jpg)|$(result_out_7)|$(result_true_7)|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/8_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/8_img.jpg)|$(result_out_8)|$(result_true_8)|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/9_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/9_img.jpg)|$(result_out_9)|$(result_true_9)|
