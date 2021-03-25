import argparse
import os
import os.path as osp

import cv2
import torch
from mmcv import Config
from torchvision.transforms import ToTensor

from components.layers import disp_to_depth
from models import RNWNet
from utils import save_disp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('sample', type=str, help='path of sample.')
    parser.add_argument('checkpoint', type=str, help='name of checkpoint.')
    parser.add_argument('out_dir', type=str, help='output directory.')
    return parser.parse_args()


if __name__ == '__main__':
    # parse args
    args = parse_args()
    # config
    cfg = Config.fromfile('config/cfg_common.py')
    # device
    device = torch.device('cpu')
    # model
    net = RNWNet(cfg.model)
    net.load_state_dict(torch.load('checkpoints/{}.pth'.format(args.checkpoint)))
    net.to(device)
    net.eval()
    # transform
    to_tensor = ToTensor()
    # visualization
    visualization_dir = args.out_dir
    if not os.path.exists(visualization_dir):
        os.mkdir(visualization_dir)
    # no grad
    with torch.no_grad():
        # read
        rgb = cv2.imread(args.sample)
        # to tensor
        t_rgb = to_tensor(rgb).unsqueeze(0).to(device)
        # feed into net
        #
        # Note: It seems that the output only have a half resolution of input
        #
        outputs = net({('color_aug', 0, 0): t_rgb})
        disp = outputs[("disp", 0, 0)]
        scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
        depth = depth[0, 0, :, :].numpy()
        # visualization
        scaled_disp = scaled_disp[0, 0, :, :].numpy()
        fn = osp.basename(args.sample)
        save_disp(scaled_disp, os.path.join(visualization_dir, fn[:-4]), max_p=95)
