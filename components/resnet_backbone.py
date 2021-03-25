import torch
from torchvision.models import resnet18, resnet34, resnet50


_MODEL_URLS = {
    'resnet18': '',
    'resnet34': '',
    'resnet50': ''
}


def load_pretrained_weights(name: str, map_location: str):
    assert name in _MODEL_URLS, 'Can not find corresponding path to {}.'.format(name)
    state_dict = torch.load(_MODEL_URLS[name], map_location=map_location)
    return state_dict


def resnet18_backbone(pre_trained=False, **kwargs):
    model = resnet18(pretrained=False, **kwargs)
    if pre_trained:
        state_dict = load_pretrained_weights('resnet18', 'cpu')
        model.load_state_dict(state_dict, strict=True)
    return model


def resnet34_backbone(pre_trained=False, **kwargs):
    model = resnet34(pretrained=False, **kwargs)
    if pre_trained:
        state_dict = load_pretrained_weights('resnet34', 'cpu')
        model.load_state_dict(state_dict, strict=True)
    return model


def resnet50_backbone(pre_trained=False, **kwargs):
    model = resnet50(pretrained=False, **kwargs)
    if pre_trained:
        state_dict = load_pretrained_weights('resnet50', 'cpu')
        model.load_state_dict(state_dict, strict=True)
    return model
