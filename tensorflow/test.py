import numpy as np
import cv2
import os
import subprocess
import glob
import tensorflow as tf
from options.test_options import TestOptions
from util.util import generate_mask_rect, generate_mask_stroke
from net.network import GMCNNModel

os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax([int(x.split()[2]) for x in subprocess.Popen(
        "nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]
        ))

config = TestOptions().parse()

if os.path.isfile(config.dataset_path):
    pathfile = open(config.dataset_path, 'rt').read().splitlines()
elif os.path.isdir(config.dataset_path):
    pathfile = glob.glob(os.path.join(config.dataset_path, '*.png'))
else:
    print('Invalid testing data file/folder path.')
    exit(1)
total_number = len(pathfile)
test_num = total_number if config.test_num == -1 else min(total_number, config.test_num)
print('The total number of testing images is {}, and we take {} for test.'.format(total_number, test_num))

model = GMCNNModel()

reuse = False
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = False
with tf.Session(config=sess_config) as sess:
    input_image_tf = tf.placeholder(dtype=tf.float32, shape=[1, config.img_shapes[0], config.img_shapes[1], 3])
    input_mask_tf = tf.placeholder(dtype=tf.float32, shape=[1, config.img_shapes[0], config.img_shapes[1], 1])

    output = model.evaluate(input_image_tf, input_mask_tf, config=config, reuse=reuse)
    output = (output + 1) * 127.5
    output = tf.minimum(tf.maximum(output[:, :, :, ::-1], 0), 255)
    output = tf.cast(output, tf.uint8)

    # load pretrained model
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = list(map(lambda x: tf.assign(x, tf.contrib.framework.load_variable(config.load_model_dir, x.name)),
                          vars_list))
    sess.run(assign_ops)
    print('Model loaded.')
    total_time = 0

    if config.random_mask:
        np.random.seed(config.seed)

    for i in range(test_num):
        if config.mask_type == 'rect':
            mask = generate_mask_rect(config.img_shapes, config.mask_shapes, config.random_mask)
        else:
            mask = generate_mask_stroke(im_size=(config.img_shapes[0], config.img_shapes[1]),
                                        parts=8, maxBrushWidth=24, maxLength=100, maxVertex=20)
        image = cv2.imread(pathfile[i])

        h, w = image.shape[:2]

        if h >= config.img_shapes[0] and w >= config.img_shapes[1]:
            h_start = (h-config.img_shapes[0]) // 2
            w_start = (w-config.img_shapes[1]) // 2
            image = image[h_start: h_start+config.img_shapes[0], w_start: w_start+config.img_shapes[1], :]
        else:
            t = min(h, w)
            image = image[(h-t)//2:(h-t)//2+t, (w-t)//2:(w-t)//2+t, :]
            image = cv2.resize(image, (config.img_shapes[1], config.img_shapes[0]))

        # cv2.imwrite(os.path.join(config.saving_path, 'gt_{:03d}.png'.format(i)), image.astype(np.uint8))
        image = image * (1-mask) + 255 * mask
        cv2.imwrite(os.path.join(config.saving_path, 'input_{:03d}.png'.format(i)), image.astype(np.uint8))

        assert image.shape[:2] == mask.shape[:2]

        h, w = image.shape[:2]
        grid = 4
        image = image[:h // grid * grid, :w // grid * grid, :]
        mask = mask[:h // grid * grid, :w // grid * grid, :]

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)

        result = sess.run(output, feed_dict={input_image_tf: image, input_mask_tf: mask})
        cv2.imwrite(os.path.join(config.saving_path, '{:03d}.png'.format(i)), result[0][:, :, ::-1])
        print(' > {} / {}'.format(i+1, test_num))
print('done.')
