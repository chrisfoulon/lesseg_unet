import logging
from typing import Tuple

from monai.networks.nets import SwinUNETR, UNETR, UNet
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
    'img_size': (96, 96, 96),
    'feature_size': 32,
    'hidden_size': 768,
    'mlp_dim': 3072,
    'num_heads': 12,
    'pos_embed': 'perceptron',
    'norm_name': 'instance',
    'res_block': True,
    'dropout_rate': 0.0,
}

default_swinunetr_hyper_params = {
    'in_channels': 1,
    'out_channels': 1,
    'img_size': (96, 96, 96),
    'feature_size': 36,
    'attn_drop_rate': 0.0,
    'dropout_path_rate': 0.0,
    'use_checkpoint': False,
    'drop_rate': 0.0,
}


def create_swinunetr_model(device: torch.device,
                           hyper_params: dict = None) -> Tuple[SwinUNETR, dict]:
    if hyper_params is None:
        hyper_params = default_swinunetr_hyper_params
    # create UNet
    logging.info('Creating monai SWIN-UNETR with params: {}'.format(hyper_params))
    model = SwinUNETR(**hyper_params).to(device)
    return model, hyper_params


def create_unetr_model(device: torch.device,
                       unetr_hyper_params: dict = None) -> Tuple[UNETR, dict]:
    if unetr_hyper_params is None:
        unetr_hyper_params = default_unetr_hyper_params
    # create UNet
    logging.info('Creating monai UNETR with params: {}'.format(unetr_hyper_params))
    model = UNETR(**unetr_hyper_params).to(device)
    return model, unetr_hyper_params


def create_unet_model(device: torch.device,
                      unet_hyper_params: dict = None) -> Tuple[UNet, dict]:
    if unet_hyper_params is None:
        unet_hyper_params = default_unet_hyper_params
    # create UNet
    logging.info('Creating monai UNet with params: {}'.format(unet_hyper_params))
    model = UNet(**unet_hyper_params).to(device)
    return model, unet_hyper_params


def create_model(device: torch.device, hyper_params: dict = None,
                 model_class_name: str = 'UNETR') -> Tuple[torch.nn.Module, dict]:
    if model_class_name.lower() == 'unetr':
        model, hyper_params = create_unetr_model(device, hyper_params)
    elif model_class_name.lower() == 'swinunetr':
        model, hyper_params = create_swinunetr_model(device, hyper_params)
    elif model_class_name.lower() == 'unet':
        model, hyper_params = create_unet_model(device, hyper_params)
    else:
        raise ValueError(f'{model_class_name} is not yet available')
    return model, hyper_params

