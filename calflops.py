from __future__ import print_function
import torchvision.models as models
import torch
import argparse
import os
import sys
sys.path.append('../')
# from resnet_1d import ResNet50_1d
# from resnet_1d_lite import ResNet50_1d_shrink
from thop import profile
import yaml
import wdsr_b
from option2 import parser
from wdsr_b import *
#from args import *
import math
# parser = argparse.ArgumentParser(description='Load Models')
# parser.add_argument('--slice_size', type=int, default=198, help='input size')
# parser.add_argument('--devices', type=int, default=500, help='number of classes')
import captioning.utils.opts as opts
import captioning.models as models
from captioning.data.dataloader import *
import numpy as np

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
opt.caption_model == 'newfc'
opt.vocab = vocab
model = models.setup(opt)
cocotest_bu_fc = np.load('~/github_yifan/ImageCaptioning.pytorch/data/cocotest_bu_fc/36184.npy')
#model = torchvision.models.alexnet()
# calculate model FLOPs
model.train(False)
model.eval()
total_flops = tu.get_model_flops(model, cocotest_bu_fc)
print('Total model FLOPs: {:,}'.format(total_flops))


# calculate total model parameters
total_params = tu.get_model_param_count(model)
print('Total model params: {:,}'.format(total_params))





