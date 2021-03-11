from math import radians

high_prob = .2
low_prob = .1
tiny_prob = 0.05
# high_prob = 1
# low_prob = 1
# tiny_prob = 1
def_spatial_size = [96, 128, 96]

# def_spatial_size = [96, 96, 96]
min_small_crop_size = [int(0.95 * d) for d in def_spatial_size]

full_hyper_dict = {
    'first_transform': [
        {'LoadImaged': {'keys': ['image', 'label']}},
        {'AddChanneld': {'keys': ['image', 'label']}},
        # {'AsChannelFirstd': {
        #     'keys': ['image', 'label'],
        #     'channel_dim': -1}
        # },
        {'Resized': {
            'keys': ['image', 'label'],
            'spatial_size': def_spatial_size,
            'mode': 'nearest'}
         },
        {'NormalizeIntensityd': {'keys': ['image']}},
        {'Binarized': {'keys': ['label'], 'lower_threshold': 0.5}},
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'Fisrt resize'}},
    ],
    'monai_transform': [
        # {'ScaleIntensity': {}}
        # {'PrintDim': {'keys': ['image', 'label']}},
        {'RandSpatialCropd': {'keys': ['image', 'label'],
                              'roi_size': min_small_crop_size,
                              'random_center': True,
                              'random_size': False}
         },
        {'RandHistogramShiftd': {
            'keys': ['image'],
            'num_control_points': (10, 15),
            'prob': low_prob}
         },
        # TODO maybe 'Orientation': {} but it would interact with the flip,
        {'RandAffined': {
            'keys': ['image', 'label'],
            'prob': high_prob,
            'rotate_range': radians(5),
            'shear_range': None,
            'translate_range': None,
            'scale_range': 0.3,
            'spatial_size': None,
            'padding_mode': 'border',
            'as_tensor_output': False}
         },
        # TODO check
        # {'RandFlipd': {
        #     'keys': ['image', 'label'],
        #     'prob': low_prob,
        #     'spatial_axis': 0}
        # },
        # {'RandDeformGrid':
        #     {'keys': ['image', 'label']}
        # },
        # {'Spacingd':
        #     {'keys': ['image', 'label']}
        # },
        {'Rand3DElasticd': {
            'keys': ['image', 'label'],
            'sigma_range': (1, 3),
            'magnitude_range': (3, 5),  # hyper_params['Rand3DElastic_magnitude_range']
            'prob': tiny_prob,
            'rotate_range': None,
            'shear_range': None,
            'translate_range': None,
            'scale_range': None,
            'spatial_size': None,
            'padding_mode': "reflection",
            # 'padding_mode': "border",
            # 'padding_mode': "zeros",
            'as_tensor_output': False}
         },
        # {'SqueezeDimd':
        #     {'keys': ["image", "label"],
        #      'dim': 0}
        # },
        {'ToTensord': {'keys': ['image', 'label']}},
        # 'AddChanneld': {'keys': ['image', 'label']},
        # 'PrintDim': {'keys': ['image', 'label'], 'msg': 'After MONAI'},
    ],
    'torchio_transform': [
        # 'PrintDim': {'keys': ['image', 'label']},
        {'RandomNoise': {
            'include': ['image'],
            'mean': 0,
            'std': (0.01, 0.1),
            'p': low_prob}
         },
        # 'RandomGhosting': {
        #     'include': ['image'],
        #     'p': tiny_prob,
        #     'num_ghosts': (4, 10)
        # },
        {'RandomBlur': {
            'include': ['image', 'label'],
            'std': (0.1, 0.5),
            'p': low_prob}
         },
        {'RandomBiasField': {
            'include': ['image'],
            'p': tiny_prob,
            'coefficients': 0.5}
         },
        {'RandomMotion': {
            'include': ['image', 'label'],
            'p': tiny_prob,
            'num_transforms': 1}
         },
        {'ToTensord': {'keys': ['image', 'label']}},
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'After TORCHIO'}},
        # {'SqueezeDimd': {'keys': ["image", "label"],
        #                 'dim': 0}},
    ],
    'labelonly_transform': [
        # {'ToTensord': {'keys': ['label']}},
        # {'AddChanneld': {'keys': ['label']}},
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'after binarize'}},
    ],
    'last_transform': [
        {'Binarized': {
            'keys': ['label'],
            'lower_threshold': 0.5}
         },
        {'Resized': {
            'keys': ['image', 'label'],
            'spatial_size': def_spatial_size,
            'mode': 'nearest'}
         },
        # 'ToTensord': {'keys': ['image', 'label']},

        # 'PrintDim': {'keys': ['image', 'label'], 'msg': 'after binarize and resize'},
        # 'AddChanneld': {'keys': ['image']},
        # 'SqueezeDimd': {'keys': ["image", "label"],
        #                 'dim': 0},
        {'NormalizeIntensityd': {'keys': ['image']}},
    ]
}


minimal_hyper_dict = {
    'first_transform': [
        {'LoadImaged': {'keys': ['image', 'label']}},
        {'AddChanneld': {'keys': ['image', 'label']}},
        # {'AsChannelFirstd': {
        #     'keys': ['image', 'label'],
        #     'channel_dim': -1}
        # },
        {'Resized': {
            'keys': ['image', 'label'],
            'spatial_size': def_spatial_size,
            # 'mode': 'nearest'
        }},
        # {'NormalizeIntensityd': {'keys': ['image']}},
        {'Binarized': {'keys': ['label'], 'lower_threshold': 0.5}},
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'First resize'}},
    ],
    'monai_transform': [

        {'ScaleIntensityd': {'keys': "image"}},

        {'ToTensord': {'keys': ['image', 'label']}},
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'Fisrt monai'}},
        # {'RandCropByPosNegLabeld': {
        #     'keys': ["image", "label"],
        #     'label_key': "label",
        #     'spatial_size': def_spatial_size,
        #     'pos': 1,
        #     'neg': 1,
        #     'num_samples': 4
        # }},
        {'RandSpatialCropd': {
            'keys': ["image", "label"],
            'roi_size': min_small_crop_size,
            'random_size': False
        }},
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'After RandCrop'}},
    ],
    'labelonly_transform': [],
    'last_transform': [
        {'Binarized': {
            'keys': ['label'],
            'lower_threshold': 0.5
        }},
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'Last binarized'}},
        {'Resized': {
            'keys': ['image', 'label'],
            'spatial_size': def_spatial_size,
            # 'mode': 'nearest'
        }},
        {'NormalizeIntensityd': {'keys': ['image']}},
    ]
}
