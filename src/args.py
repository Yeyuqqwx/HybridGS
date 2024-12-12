from arguments.arguments import args
import sys
import argparse
from datetime import datetime
import torch

torch.set_default_device("cuda:0")

args.model_path = './output/corner'
args.dataset.resolution = 1

parse = argparse.ArgumentParser()
parse.add_argument('--scene', type=str, default='None')
parse.add_argument('--source_path', type=str, default='None')
parse.add_argument('--resolution', type=int, default=1)
parse.add_argument('--ckpt_path', type=str, default='None')
parse.add_argument('--metric_only', type=str, default='False')


options = parse.parse_args()
now = datetime.now().strftime("%Y%m%d-%H%M")
if options.ckpt_path != 'None':
    options.scene = options.ckpt_path.split('/')[-1]
    
args.dataset.resolution = options.resolution

if options.ckpt_path:
    args.model_path = options.ckpt_path
else:
    args.model_path = f'../output/{options.scene}/{now}'

args.dataset.source_path = options.source_path

print(f'\033[94mSave path: {args.model_path}\033[0m')

args.overwrite_model = True
args.dataset.eval = True
####### experiments
args.dataset.white_background = True
args.dataset.images = "images"

args.gs3d.embed = 'gaussianimage'
args.gs3d.dim_embed = 10000