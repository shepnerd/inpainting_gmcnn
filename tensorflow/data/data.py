import tensorflow as tf

class DataLoader:
    def __init__(self, filename, im_size, batch_size):
        self.filelist = open(filename, 'rt').read().splitlines()
        self.im_size = im_size
        self.batch_size = batch_size
        self.data_queue = None

    def next(self):
        with tf.variable_scope('feed'):
            filelist_tensor = tf.convert_to_tensor(self.filelist, dtype=tf.string)
            self.data_queue = tf.train.slice_input_producer([filelist_tensor])

            im_gt = tf.image.decode_image(tf.read_file(self.data_queue[0]), channels=3)
            # im_gt = tf.cast(im_gt, tf.float32) / 127.5 - 1
            im_gt = tf.cast(im_gt, tf.float32)
            im_gt = tf.image.resize_image_with_crop_or_pad(im_gt, self.im_size[0], self.im_size[1])
            im_gt.set_shape([self.im_size[0], self.im_size[1], 3])
            batch_gt = tf.train.batch([im_gt], batch_size=self.batch_size, num_threads=4)
        return batch_gt
