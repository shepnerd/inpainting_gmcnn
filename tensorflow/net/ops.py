import cv2
import numpy as np
import tensorflow as tf
from logging import exception
import math
import scipy.stats as st
import os
import urllib
import scipy
from scipy import io

np.random.seed(2018)

"""
https://arxiv.org/abs/1806.03589
"""
def np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
    mask = np.zeros((h, w, 1), np.float32)
    numVertex = np.random.randint(maxVertex + 1)
    startY = np.random.randint(h)
    startX = np.random.randint(w)
    brushWidth = 0
    for i in range(numVertex):
        angle = np.random.randint(maxAngle + 1)
        angle = angle / 360.0 * 2 * np.pi
        if i % 2 == 0:
            angle = 2 * np.pi - angle
        length = np.random.randint(maxLength + 1)
        brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
        nextY = startY + length * np.cos(angle)
        nextX = startX + length * np.sin(angle)

        nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int)
        nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int)

        cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)

        startY, startX = nextY, nextX
    cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
    return mask


def free_form_mask_tf(parts, maxVertex=16, maxLength=60, maxBrushWidth=14, maxAngle=360, im_size=(256, 256), name='fmask'):
    # mask = np.zeros((im_size[0], im_size[1], 1), dtype=np.float32)
    with tf.variable_scope(name):
        mask = tf.Variable(tf.zeros([1, im_size[0], im_size[1], 1]), name='free_mask')
        maxVertex = tf.constant(maxVertex, dtype=tf.int32)
        maxLength = tf.constant(maxLength, dtype=tf.int32)
        maxBrushWidth = tf.constant(maxBrushWidth, dtype=tf.int32)
        maxAngle = tf.constant(maxAngle, dtype=tf.int32)
        h = tf.constant(im_size[0], dtype=tf.int32)
        w = tf.constant(im_size[1], dtype=tf.int32)

        p = tf.py_func(np_free_form_mask, [maxVertex, maxLength, maxBrushWidth, maxAngle, h, w], tf.float32)
        p = tf.reshape(p, [1, im_size[0], im_size[1], 1])
        for i in range(parts):
            mask = mask + p
        mask = tf.minimum(mask, 1.0)
    return mask


def gauss_kernel(size=21, sigma=3):
    interval = (2 * sigma + 1.0) / size
    x = np.linspace(-sigma-interval/2, sigma+interval/2, size+1)
    ker1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(ker1d, ker1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((size, size, 1, 1))
    return out_filter


def tf_make_guass_var(size, sigma):
    kernel = gauss_kernel(size, sigma)
    var = tf.Variable(tf.convert_to_tensor(kernel))
    return var


def priority_loss_mask(mask, hsize=64, sigma=1/40, iters=7):
    kernel = tf_make_guass_var(hsize, sigma)
    init = 1-mask
    mask_priority = None
    for i in range(iters):
        mask_priority = tf.nn.conv2d(init, kernel, strides=[1,1,1,1], padding='SAME')
        mask_priority = mask_priority * mask
        init = mask_priority + (1-mask)
    return mask_priority


def random_interpolates(x, y, alpha=None):
    """
    x: first dimension as batch_size
    y: first dimension as batch_size
    alpha: [BATCH_SIZE, 1]
    """
    shape = x.get_shape().as_list()
    x = tf.reshape(x, [shape[0], -1])
    y = tf.reshape(y, [shape[0], -1])
    if alpha is None:
        alpha = tf.random_uniform(shape=[shape[0], 1])
    interpolates = x + alpha*(y - x)
    return tf.reshape(interpolates, shape)


def random_bbox(config):
    """Generate a random tlhw with configuration.

    Args:
        config: Config should have configuration including IMG_SHAPES,
            VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.

    Returns:
        tuple: (top, left, height, width)

    """
    img_shape = config.img_shapes
    img_height = img_shape[0]
    img_width = img_shape[1]
    if config.random_mask is True:
        maxt = img_height - config.margins[0] - config.mask_shapes[0]
        maxl = img_width - config.margins[1] - config.mask_shapes[1]
        t = tf.random_uniform(
            [], minval=config.margins[0], maxval=maxt, dtype=tf.int32)
        l = tf.random_uniform(
            [], minval=config.margins[1], maxval=maxl, dtype=tf.int32)
    else:
        t = config.mask_shapes[0]//2
        l = config.mask_shapes[1]//2
    h = tf.constant(config.mask_shapes[0])
    w = tf.constant(config.mask_shapes[1])
    return (t, l, h, w)


def bbox2mask(bbox, config, name='mask'):
    """Generate mask tensor from bbox.

    Args:
        bbox: configuration tuple, (top, left, height, width)
        config: Config should have configuration including IMG_SHAPES,
            MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.

    Returns:
        tf.Tensor: output with shape [1, H, W, 1]

    """
    def npmask(bbox, height, width, delta_h, delta_w):
        mask = np.zeros((1, height, width, 1), np.float32)
        h = np.random.randint(delta_h//2+1)
        w = np.random.randint(delta_w//2+1)
        mask[:, bbox[0]+h:bbox[0]+bbox[2]-h,
             bbox[1]+w:bbox[1]+bbox[3]-w, :] = 1.
        return mask
    with tf.variable_scope(name), tf.device('/cpu:0'):
        img_shape = config.img_shapes
        height = img_shape[0]
        width = img_shape[1]
        mask = tf.py_func(
            npmask,
            [bbox, height, width,
             config.max_delta_shapes[0], config.max_delta_shapes[1]],
            tf.float32, stateful=False)
        mask.set_shape([1] + [height, width] + [1])
    return mask


def local_patch(x, bbox):
    """Crop local patch according to bbox.

    Args:
        x: input
        bbox: (top, left, height, width)

    Returns:
        tf.Tensor: local patch

    """
    x = tf.image.crop_to_bounding_box(x, bbox[0], bbox[1], bbox[2], bbox[3])
    return x


def gradients_penalty(x, y, mask=None, norm=1.):
    """Improved Training of Wasserstein GANs

    - https://arxiv.org/abs/1704.00028
    """
    gradients = tf.gradients(y, x)[0]
    if mask is None:
        mask = tf.ones_like(gradients)
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients) * mask, axis=[1, 2, 3]))
    return tf.reduce_mean(tf.square(slopes - norm))


def gan_wgan_loss(pos, neg, name='gan_wgan_loss'):
    """
    wgan loss function for GANs.

    - Wasserstein GAN: https://arxiv.org/abs/1701.07875
    """
    with tf.variable_scope(name):
        d_loss = tf.reduce_mean(neg-pos)
        g_loss = -tf.reduce_mean(neg)
        tf.summary.scalar('d_loss', d_loss)
        tf.summary.scalar('g_loss', g_loss)
        tf.summary.scalar('pos_value_avg', tf.reduce_mean(pos))
        tf.summary.scalar('neg_value_avg', tf.reduce_mean(neg))
    return g_loss, d_loss


def average_gradients(tower_grads):
    """ Calculate the average gradient for each shared variable across
    all towers.

    **Note** that this function provides a synchronization point
    across all towers.

    Args:
        tower_grads: List of lists of (gradient, variable) tuples.
            The outer list is over individual gradients. The inner list is
            over the gradient calculation for each tower.

    Returns:
        List of pairs of (gradient, variable) where the gradient
            has been averaged across all towers.

    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        v = grad_and_vars[0][1]
        # sum
        grad = tf.add_n([x[0] for x in grad_and_vars])
        # average
        grad = grad / float(len(tower_grads))
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


"""
id-mrf
"""
from enum import Enum

class Distance(Enum):
    L2 = 0
    DotProduct = 1

class CSFlow:
    def __init__(self, sigma=float(0.1), b=float(1.0)):
        self.b = b
        self.sigma = sigma

    def __calculate_CS(self, scaled_distances, axis_for_normalization=3):
        self.scaled_distances = scaled_distances
        self.cs_weights_before_normalization = tf.exp((self.b - scaled_distances) / self.sigma, name='weights_before_normalization')
        self.cs_NHWC = CSFlow.sum_normalize(self.cs_weights_before_normalization, axis_for_normalization)

    def reversed_direction_CS(self):
        cs_flow_opposite = CSFlow(self.sigma, self.b)
        cs_flow_opposite.raw_distances = self.raw_distances
        work_axis = [1, 2]
        relative_dist = cs_flow_opposite.calc_relative_distances(axis=work_axis)
        cs_flow_opposite.__calculate_CS(relative_dist, work_axis)
        return cs_flow_opposite

    # --
    @staticmethod
    def create_using_L2(I_features, T_features, sigma=float(0.1), b=float(1.0)):
        cs_flow = CSFlow(sigma, b)
        with tf.name_scope('CS'):
            sT = T_features.shape.as_list()
            sI = I_features.shape.as_list()

            Ivecs = tf.reshape(I_features, (sI[0], -1, sI[3]))
            Tvecs = tf.reshape(T_features, (sI[0], -1, sT[3]))
            r_Ts = tf.reduce_sum(Tvecs * Tvecs, 2)
            r_Is = tf.reduce_sum(Ivecs * Ivecs, 2)
            raw_distances_list = []
            for i in range(sT[TensorAxis.N]):
                Ivec, Tvec, r_T, r_I = Ivecs[i], Tvecs[i], r_Ts[i], r_Is[i]
                A = tf.matmul(Tvec,tf.transpose(Ivec))
                cs_flow.A = A
                # A = tf.matmul(Tvec, tf.transpose(Ivec))
                r_T = tf.reshape(r_T, [-1, 1])  # turn to column vector
                dist = r_T - 2 * A + r_I
                cs_shape = sI[:3] + [dist.shape[0].value]
                cs_shape[0] = 1
                dist = tf.reshape(tf.transpose(dist), cs_shape)
                # protecting against numerical problems, dist should be positive
                dist = tf.maximum(float(0.0), dist)
                # dist = tf.sqrt(dist)
                raw_distances_list += [dist]

            cs_flow.raw_distances = tf.convert_to_tensor([tf.squeeze(raw_dist, axis=0) for raw_dist in raw_distances_list])

            relative_dist = cs_flow.calc_relative_distances()
            cs_flow.__calculate_CS(relative_dist)
            return cs_flow

    #--
    @staticmethod
    def create_using_dotP(I_features, T_features, sigma=float(1.0), b=float(1.0)):
        cs_flow = CSFlow(sigma, b)
        with tf.name_scope('CS'):
            # prepare feature before calculating cosine distance
            T_features, I_features = cs_flow.center_by_T(T_features, I_features)
            with tf.name_scope('TFeatures'):
                T_features = CSFlow.l2_normalize_channelwise(T_features)
            with tf.name_scope('IFeatures'):
                I_features = CSFlow.l2_normalize_channelwise(I_features)
                # work seperatly for each example in dim 1
                cosine_dist_l = []
                N, _, _, _ = T_features.shape.as_list()
                for i in range(N):
                    T_features_i = tf.expand_dims(T_features[i, :, :, :], 0)
                    I_features_i = tf.expand_dims(I_features[i, :, :, :], 0)
                    patches_i = cs_flow.patch_decomposition(T_features_i)
                    cosine_dist_i = tf.nn.conv2d(I_features_i, patches_i, strides=[1, 1, 1, 1],
                                                        padding='VALID', use_cudnn_on_gpu=True, name='cosine_dist')
                    cosine_dist_l.append(cosine_dist_i)

                cs_flow.cosine_dist = tf.concat(cosine_dist_l, axis = 0)

                cosine_dist_zero_to_one = -(cs_flow.cosine_dist - 1) / 2
                cs_flow.raw_distances = cosine_dist_zero_to_one

                relative_dist = cs_flow.calc_relative_distances()
                cs_flow.__calculate_CS(relative_dist)
                return cs_flow

    def calc_relative_distances(self, axis=3):
        epsilon = 1e-5
        div = tf.reduce_min(self.raw_distances, axis=axis, keep_dims=True)
        # div = tf.reduce_mean(self.raw_distances, axis=axis, keep_dims=True)
        relative_dist = self.raw_distances / (div + epsilon)
        return relative_dist

    def weighted_average_dist(self, axis=3):
        if not hasattr(self, 'raw_distances'):
            raise exception('raw_distances property does not exists. cant calculate weighted average l2')

        multiply = self.raw_distances * self.cs_NHWC
        return tf.reduce_sum(multiply, axis=axis, name='weightedDistPerPatch')

    # --
    @staticmethod
    def create(I_features, T_features, distance : Distance, nnsigma=float(1.0), b=float(1.0)):
        if distance.value == Distance.DotProduct.value:
            cs_flow = CSFlow.create_using_dotP(I_features, T_features, nnsigma, b)
        elif distance.value == Distance.L2.value:
            cs_flow = CSFlow.create_using_L2(I_features, T_features, nnsigma, b)
        else:
            raise "not supported distance " + distance.__str__()
        return cs_flow

    @staticmethod
    def sum_normalize(cs, axis=3):
        reduce_sum = tf.reduce_sum(cs, axis, keep_dims=True, name='sum')
        return tf.divide(cs, reduce_sum, name='sumNormalized')

    def center_by_T(self, T_features, I_features):
        # assuming both input are of the same size

        # calculate stas over [batch, height, width], expecting 1x1xDepth tensor
        axes = [0, 1, 2]
        self.meanT, self.varT = tf.nn.moments(
            T_features, axes, name='TFeatures/moments')
        # we do not divide by std since its causing the histogram
        # for the final cs to be very thin, so the NN weights
        # are not distinctive, giving similar values for all patches.
        # stdT = tf.sqrt(varT, "stdT")
        # correct places with std zero
        # stdT[tf.less(stdT, tf.constant(0.001))] = tf.constant(1)
        with tf.name_scope('TFeatures/centering'):
            self.T_features_centered = T_features - self.meanT
        with tf.name_scope('IFeatures/centering'):
            self.I_features_centered = I_features - self.meanT

        return self.T_features_centered, self.I_features_centered

    @staticmethod
    def l2_normalize_channelwise(features):
        norms = tf.norm(features, ord='euclidean', axis=3, name='norm')
        # expanding the norms tensor to support broadcast division
        norms_expanded = tf.expand_dims(norms, 3)
        features = tf.divide(features, norms_expanded, name='normalized')
        return features

    def patch_decomposition(self, T_features):
        # patch decomposition
        patch_size = 1
        patches_as_depth_vectors = tf.extract_image_patches(
            images=T_features, ksizes=[1, patch_size, patch_size, 1],
            strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID',
            name='patches_as_depth_vectors')

        self.patches_NHWC = tf.reshape(
            patches_as_depth_vectors,
            shape=[-1, patch_size, patch_size, patches_as_depth_vectors.shape[3].value],
            name='patches_PHWC')

        self.patches_HWCN = tf.transpose(
            self.patches_NHWC,
            perm=[1, 2, 3, 0],
            name='patches_HWCP')  # tf.conv2 ready format

        return self.patches_HWCN


def mrf_loss(T_features, I_features, distance=Distance.DotProduct, nnsigma=float(1.0)):
    T_features = tf.convert_to_tensor(T_features, dtype=tf.float32)
    I_features = tf.convert_to_tensor(I_features, dtype=tf.float32)

    with tf.name_scope('cx'):
        cs_flow = CSFlow.create(I_features, T_features, distance, nnsigma)
        # sum_normalize:
        height_width_axis = [1, 2]
        # To:
        cs = cs_flow.cs_NHWC
        k_max_NC = tf.reduce_max(cs, axis=height_width_axis)
        CS = tf.reduce_mean(k_max_NC, axis=[1])
        CS_as_loss = 1 - CS
        CS_loss = -tf.log(1 - CS_as_loss)
        CS_loss = tf.reduce_mean(CS_loss)
        return CS_loss


def random_sampling(tensor_in, n, indices=None):
    N, H, W, C = tf.convert_to_tensor(tensor_in).shape.as_list()
    S = H * W
    tensor_NSC = tf.reshape(tensor_in, [N, S, C])
    all_indices = list(range(S))
    shuffled_indices = tf.random_shuffle(all_indices)
    indices = tf.gather(shuffled_indices, list(range(n)), axis=0) if indices is None else indices
    res = tf.gather(tensor_NSC, indices, axis=1)
    return res, indices


def random_pooling(feats, output_1d_size=100):
    is_input_tensor = type(feats) is tf.Tensor

    if is_input_tensor:
        feats = [feats]

    # convert all inputs to tensors
    feats = [tf.convert_to_tensor(feats_i) for feats_i in feats]

    N, H, W, C = feats[0].shape.as_list()
    feats_sampled_0, indices = random_sampling(feats[0], output_1d_size ** 2)
    res = [feats_sampled_0]
    for i in range(1, len(feats)):
        feats_sampled_i, _ = random_sampling(feats[i], -1, indices)
        res.append(feats_sampled_i)

    res = [tf.reshape(feats_sampled_i, [N, output_1d_size, output_1d_size, C]) for feats_sampled_i in res]
    if is_input_tensor:
        return res[0]
    return res


def crop_quarters(feature_tensor):
    N, fH, fW, fC = feature_tensor.shape.as_list()
    quarters_list = []
    quarter_size = [N, round(fH / 2), round(fW / 2), fC]
    quarters_list.append(tf.slice(feature_tensor, [0, 0, 0, 0], quarter_size))
    quarters_list.append(tf.slice(feature_tensor, [0, round(fH / 2), 0, 0], quarter_size))
    quarters_list.append(tf.slice(feature_tensor, [0, 0, round(fW / 2), 0], quarter_size))
    quarters_list.append(tf.slice(feature_tensor, [0, round(fH / 2), round(fW / 2), 0], quarter_size))
    feature_tensor = tf.concat(quarters_list, axis=0)
    return feature_tensor


def id_mrf_reg_feat(feat_A, feat_B, config):
    if config.crop_quarters is True:
        feat_A = crop_quarters(feat_A)
        feat_B = crop_quarters(feat_B)

    N, fH, fW, fC = feat_A.shape.as_list()
    if fH * fW <= config.max_sampling_1d_size ** 2:
        print(' #### Skipping pooling ....')
    else:
        print(' #### pooling %d**2 out of %dx%d' % (config.max_sampling_1d_size, fH, fW))
        feat_A, feat_B = random_pooling([feat_A, feat_B], output_1d_size=config.max_sampling_1d_size)

    return mrf_loss(feat_A, feat_B, distance=config.Dist, nnsigma=config.nn_stretch_sigma)


from easydict import EasyDict as edict
# scale of im_src and im_dst: [-1, 1]
def id_mrf_reg(im_src, im_dst, config):
    vgg = Vgg19(filepath=config.vgg19_path)

    src_vgg = vgg.build_vgg19((im_src + 1) * 127.5)
    dst_vgg = vgg.build_vgg19((im_dst + 1) * 127.5, reuse=True)

    feat_style_layers = config.feat_style_layers
    feat_content_layers = config.feat_content_layers

    mrf_style_w = config.mrf_style_w
    mrf_content_w = config.mrf_content_w

    mrf_config = edict()
    mrf_config.crop_quarters = False
    mrf_config.max_sampling_1d_size = 65
    mrf_config.Dist = Distance.DotProduct
    mrf_config.nn_stretch_sigma = 0.5  # 0.1

    mrf_style_loss = [w * id_mrf_reg_feat(src_vgg[layer], dst_vgg[layer], mrf_config)
                      for layer, w in feat_style_layers.items()]
    mrf_style_loss = tf.reduce_sum(mrf_style_loss)

    mrf_content_loss = [w * id_mrf_reg_feat(src_vgg[layer], dst_vgg[layer], mrf_config)
                        for layer, w in feat_content_layers.items()]
    mrf_content_loss = tf.reduce_sum(mrf_content_loss)

    id_mrf_loss = mrf_style_loss * mrf_style_w + mrf_content_loss * mrf_content_w
    return id_mrf_loss


class Vgg19(object):
    def __init__(self, filepath=None):
        self.mean = np.array([123.6800, 116.7790, 103.9390]).reshape((1, 1, 1, 3))
        self.vgg_weights = filepath if filepath is not None else os.path.join('vgg19_weights', 'imagenet-vgg-verydeep-19.mat')
        if os.path.exists(self.vgg_weights) is False:
            self.vgg_weights = os.path.join('vgg19_weights', 'imagenet-vgg-verydeep-19.mat')
            if os.path.isdir('vgg19_weights') is False:
                os.mkdir('vgg19_weights')
            url = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat'
            print('Downloading vgg19..')
            urllib.request.urlretrieve(url, self.vgg_weights)
            print('vgg19 weights have been downloaded and stored in {}'.format(self.vgg_weights))

    def build_net(self, ntype, nin, nwb=None, name=None):
        if ntype == 'conv':
            return tf.nn.relu(tf.nn.conv2d(nin, nwb[0], strides=[1, 1, 1, 1], padding='SAME', name=name) + nwb[1])
        elif ntype == 'pool':
            return tf.nn.avg_pool(nin, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def get_weight_bias(self, vgg_layers, i):
        weights = vgg_layers[i][0][0][2][0][0]
        weights = tf.constant(weights)
        bias = vgg_layers[i][0][0][2][0][1]
        bias = tf.constant(np.reshape(bias, (bias.size)))
        return weights, bias

    def build_vgg19(self, input, reuse=False):
        with tf.variable_scope('vgg19', reuse=reuse):
            net = {}
            # vgg_rawnet = scipy.io.loadmat(self.vgg_weights)
            vgg_rawnet = io.loadmat(self.vgg_weights)
            vgg_layers = vgg_rawnet['layers'][0]
            net['input'] = input - self.mean
            net['conv1_1'] = self.build_net('conv', net['input'], self.get_weight_bias(vgg_layers, 0),
                                            name='vgg_conv1_1')
            net['conv1_2'] = self.build_net('conv', net['conv1_1'], self.get_weight_bias(vgg_layers, 2),
                                            name='vgg_conv1_2')
            net['pool1'] = self.build_net('pool', net['conv1_2'])
            net['conv2_1'] = self.build_net('conv', net['pool1'], self.get_weight_bias(vgg_layers, 5),
                                            name='vgg_conv2_1')
            net['conv2_2'] = self.build_net('conv', net['conv2_1'], self.get_weight_bias(vgg_layers, 7),
                                            name='vgg_conv2_2')
            net['pool2'] = self.build_net('pool', net['conv2_2'])
            net['conv3_1'] = self.build_net('conv', net['pool2'], self.get_weight_bias(vgg_layers, 10),
                                            name='vgg_conv3_1')
            net['conv3_2'] = self.build_net('conv', net['conv3_1'], self.get_weight_bias(vgg_layers, 12),
                                            name='vgg_conv3_2')
            net['conv3_3'] = self.build_net('conv', net['conv3_2'], self.get_weight_bias(vgg_layers, 14),
                                            name='vgg_conv3_3')
            net['conv3_4'] = self.build_net('conv', net['conv3_3'], self.get_weight_bias(vgg_layers, 16),
                                            name='vgg_conv3_4')
            net['pool3'] = self.build_net('pool', net['conv3_4'])
            net['conv4_1'] = self.build_net('conv', net['pool3'], self.get_weight_bias(vgg_layers, 19),
                                            name='vgg_conv4_1')
            net['conv4_2'] = self.build_net('conv', net['conv4_1'], self.get_weight_bias(vgg_layers, 21),
                                            name='vgg_conv4_2')
            net['conv4_3'] = self.build_net('conv', net['conv4_2'], self.get_weight_bias(vgg_layers, 23),
                                            name='vgg_conv4_3')
            net['conv4_4'] = self.build_net('conv', net['conv4_3'], self.get_weight_bias(vgg_layers, 25),
                                            name='vgg_conv4_4')
            net['pool4'] = self.build_net('pool', net['conv4_4'])
            net['conv5_1'] = self.build_net('conv', net['pool4'], self.get_weight_bias(vgg_layers, 28),
                                            name='vgg_conv5_1')
            net['conv5_2'] = self.build_net('conv', net['conv5_1'], self.get_weight_bias(vgg_layers, 30),
                                            name='vgg_conv5_2')
            net['conv5_3'] = self.build_net('conv', net['conv5_2'], self.get_weight_bias(vgg_layers, 32),
                                            name='vgg_conv5_3')
            net['conv5_4'] = self.build_net('conv', net['conv5_3'], self.get_weight_bias(vgg_layers, 34),
                                            name='vgg_conv5_4')
            net['pool5'] = self.build_net('pool', net['conv5_4'])
        return net
