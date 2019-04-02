import argparse
import os
import time

class TrainOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.dataset_training_paths = {
            'celebahq': '',
            'celeba': '',
            'places2': '',
            'imagenet': '',
            'paris_streetview': '',
            'places2full': '',
            'adobe_5k': '',
            'celeba_wild': ''
        }

    def initialize(self):
        self.parser.add_argument('--dataset', type=str, default='paris_streetview', help='The dataset of the experiment.')
        self.parser.add_argument('--data_file', type=str, default='', help='the file storing training file paths')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--load_model_dir', type=str, default='', help='pretrained models are given here')
        self.parser.add_argument('--model_prefix', type=str, default='snap', help='models are saved here')

        self.parser.add_argument('--batch_size', type=int, default=16, help='input batch size')

        self.parser.add_argument('--random_mask', type=int, default=1)
        self.parser.add_argument('--mask_type', type=str, default='rect')
        self.parser.add_argument('--pretrain_network', type=int, default=0)
        self.parser.add_argument('--gan_loss_alpha', type=float, default=1e-3)
        self.parser.add_argument('--wgan_gp_lambda', type=float, default=10)
        self.parser.add_argument('--pretrain_l1_alpha', type=float, default=1.2)
        self.parser.add_argument('--l1_loss_alpha', type=float, default=1.4)
        self.parser.add_argument('--ae_loss_alpha', type=float, default=1.2)
        self.parser.add_argument('--mrf_alpha', type=float, default=0.05)
        self.parser.add_argument('--random_seed', type=bool, default=False)
        self.parser.add_argument('--lr', type=float, default=1e-5, help='learning rate for training')

        self.parser.add_argument('--train_spe', type=int, default=1000)
        self.parser.add_argument('--max_iters', type=int, default=40000)
        self.parser.add_argument('--viz_steps', type=int, default=5)

        self.parser.add_argument('--img_shapes', type=str, default='256,256,3',
                                 help='given shape parameters: h,w,c or h,w')
        self.parser.add_argument('--mask_shapes', type=str, default='128,128',
                                 help='given mask parameters: h,w')
        self.parser.add_argument('--max_delta_shapes', type=str, default='32,32')
        self.parser.add_argument('--margins', type=str, default='0,0')
        # for generator
        self.parser.add_argument('--g_cnum', type=int, default=32,
                                 help='# of generator filters in first conv layer')
        self.parser.add_argument('--d_cnum', type=int, default=64,
                                 help='# of discriminator filters in first conv layer')

        self.parser.add_argument('--vgg19_path', type=str, default='vgg19_weights/imagenet-vgg-verydeep-19.mat')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        assert self.opt.dataset in self.dataset_training_paths.keys()
        self.opt.dataset_path = \
            self.dataset_training_paths[self.opt.dataset] if self.opt.data_file == '' else self.opt.data_file

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(str(id))

        assert self.opt.random_mask in [0, 1]
        self.opt.random_mask = True if self.opt.random_mask == 1 else False

        assert self.opt.pretrain_network in [0, 1]
        self.opt.pretrain_network = True if self.opt.pretrain_network == 1 else False

        assert self.opt.mask_type in ['rect', 'stroke']

        str_img_shapes = self.opt.img_shapes.split(',')
        self.opt.img_shapes = [int(x) for x in str_img_shapes]

        str_mask_shapes = self.opt.mask_shapes.split(',')
        self.opt.mask_shapes = [int(x) for x in str_mask_shapes]

        str_max_delta_shapes = self.opt.max_delta_shapes.split(',')
        self.opt.max_delta_shapes = [int(x) for x in str_max_delta_shapes]

        str_margins = self.opt.margins.split(',')
        self.opt.margins = [int(x) for x in str_margins]

        # model name and date
        self.opt.date_str = time.strftime('%Y%m%d-%H%M%S')
        self.opt.model_name = 'GMCNN'
        self.opt.model_folder = self.opt.date_str + '_' + self.opt.model_name
        self.opt.model_folder += '_' + self.opt.dataset
        self.opt.model_folder += '_b' + str(self.opt.batch_size)
        self.opt.model_folder += '_s' + str(self.opt.img_shapes[0]) + 'x' + str(self.opt.img_shapes[1])
        self.opt.model_folder += '_gc' + str(self.opt.g_cnum)
        self.opt.model_folder += '_dc' + str(self.opt.d_cnum)

        self.opt.model_folder += '_randmask-' + self.opt.mask_type if self.opt.random_mask else ''
        self.opt.model_folder += '_pretrain' if self.opt.pretrain_network else ''

        if os.path.isdir(self.opt.checkpoints_dir) is False:
            os.mkdir(self.opt.checkpoints_dir)

        self.opt.model_folder = os.path.join(self.opt.checkpoints_dir, self.opt.model_folder)
        if os.path.isdir(self.opt.model_folder) is False:
            os.mkdir(self.opt.model_folder)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(self.opt.gpu_ids)

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        return self.opt
