from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn

from .disp_decoder import DispDecoder
from .disp_encoder import DispEncoder
from .pose_decoder import PoseDecoder
from .pose_encoder import PoseEncoder


class RNWNet(nn.Module):
    """
    MonoDepth2
    """
    def __init__(self, options):
        super(RNWNet, self).__init__()
        self.opt = options
        self.num_input_frames = len(self.opt.frame_ids)

        # components
        self.DepthEncoder = DispEncoder(self.opt.depth_num_layers, pre_trained=False)
        self.DepthDecoder = DispDecoder(self.DepthEncoder.num_ch_enc)

        self.PoseEncoder = PoseEncoder(self.opt.pose_num_layers, False, num_input_images=2)
        self.PoseDecoder = PoseDecoder(self.PoseEncoder.num_ch_enc)

    def forward(self, inputs):
        outputs = self.DepthDecoder(self.DepthEncoder(inputs['color_aug', 0, 0]))
        if self.training:
            outputs.update(self.predict_poses(inputs))
        return outputs

    def predict_poses(self, inputs):
        outputs = {}
        pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
        for f_i in self.opt.frame_ids[1:]:
            if not f_i == "s":
                if f_i < 0:
                    pose_inputs = [pose_feats[f_i], pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0], pose_feats[f_i]]
                pose_inputs = self.PoseEncoder(torch.cat(pose_inputs, 1))
                axisangle, translation = self.PoseDecoder(pose_inputs)
                outputs[("cam_T_cam", 0, f_i)] = self.transformation_from_parameters(axisangle[:, 0], translation[:, 0],
                                                                                     invert=(f_i < 0))
        return outputs

    def transformation_from_parameters(self, axisangle, translation, invert=False):
        R = self.rot_from_axisangle(axisangle)
        t = translation.clone()
        if invert:
            R = R.transpose(1, 2)
            t *= -1
        T = self.get_translation_matrix(t)
        if invert:
            M = torch.matmul(R, T)
        else:
            M = torch.matmul(T, R)
        return M

    @staticmethod
    def get_translation_matrix(translation_vector):
        T = torch.zeros(translation_vector.shape[0], 4, 4).cuda()
        t = translation_vector.contiguous().view(-1, 3, 1)
        T[:, 0, 0] = 1
        T[:, 1, 1] = 1
        T[:, 2, 2] = 1
        T[:, 3, 3] = 1
        T[:, :3, 3, None] = t
        return T

    @staticmethod
    def rot_from_axisangle(vec):
        angle = torch.norm(vec, 2, 2, True)
        axis = vec / (angle + 1e-7)
        ca = torch.cos(angle)
        sa = torch.sin(angle)
        C = 1 - ca
        x = axis[..., 0].unsqueeze(1)
        y = axis[..., 1].unsqueeze(1)
        z = axis[..., 2].unsqueeze(1)
        xs = x * sa
        ys = y * sa
        zs = z * sa
        xC = x * C
        yC = y * C
        zC = z * C
        xyC = x * yC
        yzC = y * zC
        zxC = z * xC
        rot = torch.zeros((vec.shape[0], 4, 4)).cuda()
        rot[:, 0, 0] = torch.squeeze(x * xC + ca)
        rot[:, 0, 1] = torch.squeeze(xyC - zs)
        rot[:, 0, 2] = torch.squeeze(zxC + ys)
        rot[:, 1, 0] = torch.squeeze(xyC + zs)
        rot[:, 1, 1] = torch.squeeze(y * yC + ca)
        rot[:, 1, 2] = torch.squeeze(yzC - xs)
        rot[:, 2, 0] = torch.squeeze(zxC - ys)
        rot[:, 2, 1] = torch.squeeze(yzC + xs)
        rot[:, 2, 2] = torch.squeeze(z * zC + ca)
        rot[:, 3, 3] = 1
        return rot
