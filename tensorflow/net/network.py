import tensorflow as tf
from net.ops import random_bbox, bbox2mask, local_patch
from net.ops import priority_loss_mask
from net.ops import id_mrf_reg
from net.ops import gan_wgan_loss, gradients_penalty, random_interpolates
from net.ops import free_form_mask_tf
from util.util import f2uint
from functools import partial

class GMCNNModel:
    def __init__(self):
        self.config = None

        # shortcut ops
        self.conv7 = partial(tf.layers.conv2d, kernel_size=7, activation=tf.nn.elu, padding='SAME')
        self.conv5 = partial(tf.layers.conv2d, kernel_size=5, activation=tf.nn.elu, padding='SAME')
        self.conv3 = partial(tf.layers.conv2d, kernel_size=3, activation=tf.nn.elu, padding='SAME')
        self.conv5_ds = partial(tf.layers.conv2d, kernel_size=5, strides=2, activation=tf.nn.leaky_relu, padding='SAME')

    def build_generator(self, x, mask, reuse=False, name='inpaint_net'):
        xshape = x.get_shape().as_list()
        xh, xw = xshape[1], xshape[2]
        ones_x = tf.ones_like(x)[:, :, :, 0:1]
        x_w_mask = tf.concat([x, ones_x, ones_x * mask], axis=3)

        # network with three branches
        cnum = self.config.g_cnum
        b_names = ['b1', 'b2', 'b3', 'merge']

        conv_7 = self.conv7
        conv_5 = self.conv5
        conv_3 = self.conv3
        with tf.variable_scope(name, reuse=reuse):
            # branch 1
            x = conv_7(inputs=x_w_mask, filters=cnum, strides=1, name=b_names[0] + 'conv1')
            x = conv_7(inputs=x, filters=2*cnum, strides=2, name=b_names[0] + 'conv2_downsample')
            x = conv_7(inputs=x, filters=2*cnum, strides=1, name=b_names[0] + 'conv3')
            x = conv_7(inputs=x, filters=4*cnum, strides=2, name=b_names[0] + 'conv4_downsample')
            x = conv_7(inputs=x, filters=4*cnum, strides=1, name=b_names[0] + 'conv5')
            x = conv_7(inputs=x, filters=4*cnum, strides=1, name=b_names[0] + 'conv6')
            x = conv_7(inputs=x, filters=4*cnum, strides=1, dilation_rate=2, name=b_names[0] + 'conv7_atrous')
            x = conv_7(inputs=x, filters=4*cnum, strides=1, dilation_rate=4, name=b_names[0] + 'conv8_atrous')
            x = conv_7(inputs=x, filters=4*cnum, strides=1, dilation_rate=8, name=b_names[0] + 'conv9_atrous')
            x = conv_7(inputs=x, filters=4*cnum, strides=1, dilation_rate=16, name=b_names[0] + 'conv10_atrous')
            if cnum > 32:
                x = conv_7(inputs=x, filters=4 * cnum, strides=1, dilation_rate=32, name=b_names[0] + 'conv11_atrous')
            x = conv_7(inputs=x, filters=4*cnum, strides=1, name=b_names[0] + 'conv11')
            x = conv_7(inputs=x, filters=4*cnum, strides=1, name=b_names[0] + 'conv12')
            x_b1 = tf.image.resize_bilinear(x, [xh, xw], align_corners=True)

            # branch 2
            x = conv_5(inputs=x_w_mask, filters=cnum, strides=1, name=b_names[1] + 'conv1')
            x = conv_5(inputs=x, filters=2 * cnum, strides=2, name=b_names[1] + 'conv2_downsample')
            x = conv_5(inputs=x, filters=2 * cnum, strides=1, name=b_names[1] + 'conv3')
            x = conv_5(inputs=x, filters=4 * cnum, strides=2, name=b_names[1] + 'conv4_downsample')
            x = conv_5(inputs=x, filters=4 * cnum, strides=1, name=b_names[1] + 'conv5')
            x = conv_5(inputs=x, filters=4 * cnum, strides=1, name=b_names[1] + 'conv6')
            x = conv_5(inputs=x, filters=4 * cnum, strides=1, dilation_rate=2, name=b_names[1] + 'conv7_atrous')
            x = conv_5(inputs=x, filters=4 * cnum, strides=1, dilation_rate=4, name=b_names[1] + 'conv8_atrous')
            x = conv_5(inputs=x, filters=4 * cnum, strides=1, dilation_rate=8, name=b_names[1] + 'conv9_atrous')
            x = conv_5(inputs=x, filters=4 * cnum, strides=1, dilation_rate=16, name=b_names[1] + 'conv10_atrous')
            if cnum > 32:
                x = conv_5(inputs=x, filters=4 * cnum, strides=1, dilation_rate=32, name=b_names[1] + 'conv11_atrous')
            x = conv_5(inputs=x, filters=4 * cnum, strides=1, name=b_names[1] + 'conv11')
            x = conv_5(inputs=x, filters=4 * cnum, strides=1, name=b_names[1] + 'conv12')
            x = tf.image.resize_nearest_neighbor(x, [xh//2, xw//2], align_corners=True)
            with tf.variable_scope(b_names[1] + 'conv13_upsample'):
                x = conv_3(inputs=x, filters=2 * cnum, strides=1, name=b_names[1] + 'conv13_upsample_conv')
            x = conv_5(inputs=x, filters=2 * cnum, strides=1, name=b_names[1] + 'conv14')
            x_b2 = tf.image.resize_bilinear(x, [xh, xw], align_corners=True)

            # branch 3
            x = conv_5(inputs=x_w_mask, filters=cnum, strides=1, name=b_names[2] + 'conv1')
            x = conv_3(inputs=x, filters=2 * cnum, strides=2, name=b_names[2] + 'conv2_downsample')
            x = conv_3(inputs=x, filters=2 * cnum, strides=1, name=b_names[2] + 'conv3')
            x = conv_3(inputs=x, filters=4 * cnum, strides=2, name=b_names[2] + 'conv4_downsample')
            x = conv_3(inputs=x, filters=4 * cnum, strides=1, name=b_names[2] + 'conv5')
            x = conv_3(inputs=x, filters=4 * cnum, strides=1, name=b_names[2] + 'conv6')
            x = conv_3(inputs=x, filters=4 * cnum, strides=1, dilation_rate=2, name=b_names[2] + 'conv7_atrous')
            x = conv_3(inputs=x, filters=4 * cnum, strides=1, dilation_rate=4, name=b_names[2] + 'conv8_atrous')
            x = conv_3(inputs=x, filters=4 * cnum, strides=1, dilation_rate=8, name=b_names[2] + 'conv9_atrous')
            x = conv_3(inputs=x, filters=4 * cnum, strides=1, dilation_rate=16, name=b_names[2] + 'conv10_atrous')
            if cnum > 32:
                x = conv_3(inputs=x, filters=4 * cnum, strides=1, dilation_rate=32, name=b_names[2] + 'conv11_atrous')
            x = conv_3(inputs=x, filters=4 * cnum, strides=1, name=b_names[2] + 'conv11')
            x = conv_3(inputs=x, filters=4 * cnum, strides=1, name=b_names[2] + 'conv12')
            x = tf.image.resize_nearest_neighbor(x, [xh // 2, xw // 2], align_corners=True)
            with tf.variable_scope(b_names[2] + 'conv13_upsample'):
                x = conv_3(inputs=x, filters=2 * cnum, strides=1, name=b_names[2] + 'conv13_upsample_conv')
            x = conv_3(inputs=x, filters=2 * cnum, strides=1, name=b_names[2] + 'conv14')
            x = tf.image.resize_nearest_neighbor(x, [xh, xw], align_corners=True)
            with tf.variable_scope(b_names[2] + 'conv15_upsample'):
                x = conv_3(inputs=x, filters=cnum, strides=1, name=b_names[2] + 'conv15_upsample_conv')
            x_b3 = conv_3(inputs=x, filters=cnum//2, strides=1, name=b_names[2] + 'conv16')

            x_merge = tf.concat([x_b1, x_b2, x_b3], axis=3)

            x = conv_3(inputs=x_merge, filters=cnum // 2, strides=1, name=b_names[3] + 'conv17')
            x = tf.layers.conv2d(inputs=x, kernel_size=3, filters=3, strides=1, activation=None, padding='SAME',
                                 name=b_names[3] + 'conv18')
            x = tf.clip_by_value(x, -1., 1.)
        return x


    def wgan_patch_discriminator(self, x, mask, d_cnum, reuse=False):
        cnum = d_cnum
        with tf.variable_scope('discriminator_local', reuse=reuse):
            h, w = mask.get_shape().as_list()[1:3]
            x = self.conv5_ds(x, filters=cnum, name='conv1')
            x = self.conv5_ds(x, filters=cnum*2, name='conv2')
            x = self.conv5_ds(x, filters=cnum*4, name='conv3')
            x = self.conv5_ds(x, filters=cnum*8, name='conv4')
            x = tf.layers.conv2d(x, kernel_size=5, strides=2, filters=1, activation=None, name='conv5', padding='SAME')

            mask = tf.contrib.layers.max_pool2d(mask, 2, padding='SAME')
            mask = tf.contrib.layers.max_pool2d(mask, 2, padding='SAME')
            mask = tf.contrib.layers.max_pool2d(mask, 2, padding='SAME')
            mask = tf.contrib.layers.max_pool2d(mask, 2, padding='SAME')
            mask = tf.contrib.layers.max_pool2d(mask, 2, padding='SAME')

            x = x * mask
            x = tf.reduce_sum(x, axis=[1, 2, 3]) / tf.reduce_sum(mask, axis=[1, 2, 3])
            mask_local = tf.image.resize_nearest_neighbor(mask, [h, w], align_corners=True)
            return x, mask_local


    def wgan_local_discriminator(self, x, d_cnum, reuse=False):
        cnum = d_cnum
        with tf.variable_scope('disc_local', reuse=reuse):
            x = self.conv5_ds(x, filters=cnum, name='conv1')
            x = self.conv5_ds(x, filters=cnum * 2, name='conv2')
            x = self.conv5_ds(x, filters=cnum * 4, name='conv3')
            x = self.conv5_ds(x, filters=cnum * 8, name='conv4')
            x = self.conv5_ds(x, filters=cnum * 4, name='conv5')
            x = self.conv5_ds(x, filters=cnum * 2, name='conv6')

            x = tf.layers.flatten(x, name='flatten')
            return x

    def wgan_global_discriminator(self, x, d_cnum, reuse=False):
        cnum = d_cnum
        with tf.variable_scope('disc_global', reuse=reuse):
            x = self.conv5_ds(x, filters=cnum, name='conv1')
            x = self.conv5_ds(x, filters=cnum * 2, name='conv2')
            x = self.conv5_ds(x, filters=cnum * 4, name='conv3')
            x = self.conv5_ds(x, filters=cnum * 8, name='conv4')
            x = self.conv5_ds(x, filters=cnum * 4, name='conv5')
            x = self.conv5_ds(x, filters=cnum * 2, name='conv6')
            x = tf.layers.flatten(x, name='flatten')
            return x

    def wgan_discriminator(self, batch_local, batch_global, d_cnum, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            dlocal = self.wgan_local_discriminator(batch_local, d_cnum, reuse=reuse)
            dglobal = self.wgan_global_discriminator(batch_global, d_cnum, reuse=reuse)
            dout_local = tf.layers.dense(dlocal, 1, name='dout_local_fc')
            dout_global = tf.layers.dense(dglobal, 1, name='dout_global_fc')
            return dout_local, dout_global

    def wgan_mask_discriminator(self, batch_global, mask, d_cnum, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            dglobal = self.wgan_global_discriminator(batch_global, d_cnum, reuse=reuse)
            dout_global = tf.layers.dense(dglobal, 1, name='dout_global_fc')
            dout_local, mask_local = self.wgan_patch_discriminator(batch_global, mask, d_cnum, reuse=reuse)
        return dout_local, dout_global, mask_local

    def build_net(self, batch_data, config, summary=True, reuse=False):
        self.config = config
        batch_pos = batch_data / 127.5 - 1.
        # generate mask, 1 represents masked point
        if config.mask_type == 'rect':
            bbox = random_bbox(config)
            mask = bbox2mask(bbox, config, name='mask_c')
        else:
            mask = free_form_mask_tf(parts=8, im_size=(config.img_shapes[0], config.img_shapes[1]),
                                     maxBrushWidth=20, maxLength=80, maxVertex=16)
        batch_incomplete = batch_pos * (1. - mask)
        mask_priority = priority_loss_mask(mask)
        batch_predicted = self.build_generator(batch_incomplete, mask, reuse=reuse)

        losses = {}
        # apply mask and complete image
        batch_complete = batch_predicted * mask + batch_incomplete * (1. - mask)
        if config.mask_type == 'rect':
            # local patches
            local_patch_batch_pos = local_patch(batch_pos, bbox)
            local_patch_batch_complete = local_patch(batch_complete, bbox)
            local_patch_mask = local_patch(mask, bbox)
            local_patch_batch_pred = local_patch(batch_predicted, bbox)
            mask_priority = local_patch(mask_priority, bbox)
        else:
            local_patch_batch_pos = batch_pos
            local_patch_batch_complete = batch_complete
            local_patch_batch_pred = batch_predicted

        if config.pretrain_network:
            print('Pretrain the whole net with only reconstruction loss.')

        if not config.pretrain_network:
            config.feat_style_layers = {'conv3_2': 1.0, 'conv4_2': 1.0}
            config.feat_content_layers = {'conv4_2': 1.0}

            config.mrf_style_w = 1.0
            config.mrf_content_w = 1.0

            ID_MRF_loss = id_mrf_reg(local_patch_batch_pred, local_patch_batch_pos, config)
            # ID_MRF_loss = id_mrf_reg(batch_predicted, batch_pos, config)

            losses['ID_MRF_loss'] = ID_MRF_loss
            tf.summary.scalar('losses/ID_MRF_loss', losses['ID_MRF_loss'])

        pretrain_l1_alpha = config.pretrain_l1_alpha
        losses['l1_loss'] = \
            pretrain_l1_alpha * tf.reduce_mean(tf.abs(local_patch_batch_pos - local_patch_batch_pred) * mask_priority)
        if not config.pretrain_network:
            losses['l1_loss'] += tf.reduce_mean(ID_MRF_loss * config.mrf_alpha)
        losses['ae_loss'] = pretrain_l1_alpha * tf.reduce_mean(tf.abs(batch_pos - batch_predicted) * (1. - mask))
        if not config.pretrain_network:
            losses['ae_loss'] += pretrain_l1_alpha * tf.reduce_mean(tf.abs(batch_pos - batch_predicted) * (1. - mask))
        losses['ae_loss'] /= tf.reduce_mean(1. - mask)

        if summary:
            viz_img = tf.concat([batch_pos, batch_incomplete, batch_complete], axis=2)
            tf.summary.image('gt__degraded__predicted', f2uint(viz_img))
            tf.summary.scalar('losses/l1_loss', losses['l1_loss'])
            tf.summary.scalar('losses/ae_loss', losses['ae_loss'])

        # gan
        batch_pos_neg = tf.concat([batch_pos, batch_complete], axis=0)

        if config.mask_type == 'rect':
            # local deterministic patch
            local_patch_batch_pos_neg = tf.concat([local_patch_batch_pos, local_patch_batch_complete], 0)
            # wgan with gradient penalty
            pos_neg_local, pos_neg_global = self.wgan_discriminator(local_patch_batch_pos_neg,
                                                                    batch_pos_neg, config.d_cnum, reuse=reuse)
        else:
            pos_neg_local, pos_neg_global, mask_local = self.wgan_mask_discriminator(batch_pos_neg,
                                                                                     mask, config.d_cnum, reuse=reuse)
        pos_local, neg_local = tf.split(pos_neg_local, 2)
        pos_global, neg_global = tf.split(pos_neg_global, 2)
        # wgan loss
        global_wgan_loss_alpha = 1.0
        g_loss_local, d_loss_local = gan_wgan_loss(pos_local, neg_local, name='gan/local_gan')
        g_loss_global, d_loss_global = gan_wgan_loss(pos_global, neg_global, name='gan/global_gan')
        losses['g_loss'] = global_wgan_loss_alpha * g_loss_global + g_loss_local
        losses['d_loss'] = d_loss_global + d_loss_local
        # gp
        interpolates_global = random_interpolates(batch_pos, batch_complete)
        if config.mask_type == 'rect':
            interpolates_local = random_interpolates(local_patch_batch_pos, local_patch_batch_complete)
            dout_local, dout_global = self.wgan_discriminator(
                interpolates_local, interpolates_global, config.d_cnum, reuse=True)
        else:
            interpolates_local = interpolates_global
            dout_local, dout_global, _ = self.wgan_mask_discriminator(interpolates_global, mask, config.d_cnum, reuse=True)

        # apply penalty
        if config.mask_type == 'rect':
            penalty_local = gradients_penalty(interpolates_local, dout_local, mask=local_patch_mask)
        else:
            penalty_local = gradients_penalty(interpolates_local, dout_local, mask=mask)
        penalty_global = gradients_penalty(interpolates_global, dout_global, mask=mask)
        losses['gp_loss'] = config.wgan_gp_lambda * (penalty_local + penalty_global)
        losses['d_loss'] = losses['d_loss'] + losses['gp_loss']
        if summary and not config.pretrain_network:
            tf.summary.scalar('convergence/d_loss', losses['d_loss'])
            tf.summary.scalar('convergence/local_d_loss', d_loss_local)
            tf.summary.scalar('convergence/global_d_loss', d_loss_global)
            tf.summary.scalar('gan_wgan_loss/gp_loss', losses['gp_loss'])
            tf.summary.scalar('gan_wgan_loss/gp_penalty_local', penalty_local)
            tf.summary.scalar('gan_wgan_loss/gp_penalty_global', penalty_global)

        if config.pretrain_network:
            losses['g_loss'] = 0
        else:
            losses['g_loss'] = config.gan_loss_alpha * losses['g_loss']
        losses['g_loss'] += config.l1_loss_alpha * losses['l1_loss']
        ##

        print('Set L1_LOSS_ALPHA to %f' % config.l1_loss_alpha)
        print('Set GAN_LOSS_ALPHA to %f' % config.gan_loss_alpha)

        losses['g_loss'] += config.ae_loss_alpha * losses['ae_loss']
        print('Set AE_LOSS_ALPHA to %f' % config.ae_loss_alpha)
        g_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'inpaint_net')
        d_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        return g_vars, d_vars, losses

    def evaluate(self, im, mask, config, reuse=False):
        # generate mask, 1 represents masked point
        self.config = config
        im = im / 127.5 - 1
        im = im * (1 - mask)
        # inpaint
        batch_predict = self.build_generator(im, mask, reuse=reuse)
        # apply mask and reconstruct
        batch_complete = batch_predict * mask + im * (1 - mask)
        return batch_complete
