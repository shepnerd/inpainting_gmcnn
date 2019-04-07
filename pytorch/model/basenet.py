import os
import torch
import torch.nn as nn

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

    def init(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.save_dir = opt.checkpoint_dir
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')

    def forward(self, *input):
        return super(BaseNet, self).forward(*input)

    def test(self, *input):
        with torch.no_grad():
            self.forward(*input)

    def save_network(self, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(self.cpu().state_dict(), save_path)

    def load_network(self, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
        else:
            try:
                self.load_state_dict(torch.load(save_path))
            except:
                pretrained_dict = torch.load(save_path)
                model_dict = self.state_dict()
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    self.load_state_dict(pretrained_dict)
                    print('Pretrained network %s has excessive layers; Only loading layers that are used' % network_label)
                except:
                    print('Pretrained network %s has fewer layers; The following are not initialized: ' % network_label)
                    for k, v in pretrained_dict.items():
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v

                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            print(k.split('.')[0])
                    self.load_state_dict(model_dict)
