from __future__ import print_function
import torchvision.models as models
import torch
import argparse
import os
import sys
sys.path.append('../')
# from resnet_1d import ResNet50_1d
# from resnet_1d_lite import ResNet50_1d_shrink
#from args import *
import math
# parser = argparse.ArgumentParser(description='Load Models')
# parser.add_argument('--slice_size', type=int, default=198, help='input size')
# parser.add_argument('--devices', type=int, default=500, help='number of classes')
import captioning.utils.opts as opts
import captioning.models as models
from captioning.data.dataloader import *
import numpy as np
import captioning.utils.misc as utils
import torchvision
import torchutils as tu

# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--model', type=str, default='',
                help='path to model to evaluate')
parser.add_argument('--cnn_model', type=str,  default='resnet101',
                help='resnet101, resnet152')
parser.add_argument('--infos_path', type=str, default='',
                help='path to infos to evaluate')
parser.add_argument('--only_lang_eval', type=int, default=0,
                help='lang eval on saved results')
parser.add_argument('--force', type=int, default=0,
                help='force to evaluate no matter if there are results available')
opts.add_eval_options(parser)
opts.add_diversity_opts(parser)
opt = parser.parse_args()
opt.caption_model = 'newfc'
opt.infos_path = '/home/zzgyf/github_yifan/ImageCaptioning.pytorch/models/infos_fc_nsc-best.pkl'
with open(opt.infos_path, 'rb') as f:
    infos = utils.pickle_load(f)

replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_label_h5', 'input_json', 'batch_size', 'id']
ignore = ['start_from']

for k in vars(infos['opt']).keys():
    if k in replace:
        setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))
    elif k not in ignore:
        if not k in vars(opt):
            vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

vocab = infos['vocab'] # ix -> word mapping

opt.vocab = vocab
model = models.setup(opt)
cocotest_bu_fc = np.load('/home/zzgyf/github_yifan/ImageCaptioning.pytorch/data/cocotest_bu_fc/36184.npy')
cocotest_bu_att = np.load('/home/zzgyf/github_yifan/ImageCaptioning.pytorch/data/cocotest_bu_att/36184.npz')
#model = torchvision.models.alexnet()
# calculate model FLOPs
model.train(False)
model.eval()
total_flops = tu.get_model_flops(model, cocotest_bu_fc, cocotest_bu_att)
print('Total model FLOPs: {:,}'.format(total_flops))


# calculate total model parameters
total_params = tu.get_model_param_count(model)
print('Total model params: {:,}'.format(total_params))





