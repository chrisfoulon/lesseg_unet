import logging
from copy import deepcopy

import monai
import torch


default_unet_hyper_params = {
        'dimensions': 3,
        'in_channels': 1,
        'out_channels': 1,
        # 'channels': (16, 32, 64, 128, 256),
        'channels': (10, 20, 40, 80, 160),
        # 'strides': (2, 2, 2, 2),
        'strides': (2, 2, 2, 2, 2),
        'num_res_units': 2,
        # 'dropout': 0.5
    }

coord_conv_unet_hyper_params = deepcopy(default_unet_hyper_params)
coord_conv_unet_hyper_params['in_channels'] = 4


def create_unet_model(device: torch.device,
                      unet_hyper_params: dict = None) -> monai.networks.nets.unet:
    if unet_hyper_params is None:
        unet_hyper_params = default_unet_hyper_params
    # create UNet
    logging.info('Creating monai UNet with params: {}'.format(unet_hyper_params))
    model = monai.networks.nets.UNet(**unet_hyper_params).to(device)
    return model
