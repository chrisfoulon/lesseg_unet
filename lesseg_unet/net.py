import logging
from copy import deepcopy

from monai.networks.nets import UNETR, UNet
import torch


default_unet_hyper_params = {
        'dimensions': 3,
        'in_channels': 1,
        'out_channels': 1,
        'channels': (32, 64, 96, 128, 160),
        # 'channels': (16, 32, 64, 128, 256),
        # 'channels': (10, 20, 40, 80, 160),
        # 'strides': (2, 2, 2, 2),
        'strides': (2, 2, 2, 2),  # 2),
        'num_res_units': 2,
        'dropout': 0.0
    }
# TODO outdated
# coord_conv_unet_hyper_params = deepcopy(default_unet_hyper_params)
# coord_conv_unet_hyper_params['in_channels'] = 4

default_unetr_hyper_params = {
    'in_channels': 1,
    'out_channels': 1,
    'img_size': (32, 32, 32),
    'feature_size': 32,
    'hidden_size': 768,
    'mlp_dim': 3072,
    'num_heads': 12,
    'pos_embed': 'perceptron',
    'norm_name': 'instance',
    'res_block': True,
    'dropout_rate': 0.0,
}


def create_unetr_model(device: torch.device,
                       unetr_hyper_params: dict = None) -> UNETR:
    if unetr_hyper_params is None:
        unetr_hyper_params = default_unetr_hyper_params
    # create UNet
    logging.info('Creating monai UNETR with params: {}'.format(unetr_hyper_params))
    model = UNETR(**unetr_hyper_params).to(device)
    return model


def create_unet_model(device: torch.device,
                      unet_hyper_params: dict = None) -> UNet:
    if unet_hyper_params is None:
        unet_hyper_params = default_unet_hyper_params
    # create UNet
    logging.info('Creating monai UNet with params: {}'.format(unet_hyper_params))
    model = UNet(**unet_hyper_params).to(device)
    return model
