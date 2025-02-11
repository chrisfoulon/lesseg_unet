from math import radians
from copy import deepcopy

high_prob = .2
low_prob = .1
tiny_prob = 0.05
# high_prob = 1
# low_prob = 1
# tiny_prob = 1
def_spatial_size = [96, 128, 96]
# def_spatial_size = [64, 64, 64]
# def_spatial_size = [96, 96, 96]
# for 1mm images
# def_spatial_size = [192, 224, 192]

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
        # {'Resized': {
        #     'keys': ['image', 'label'],
        #     'spatial_size': def_spatial_size,
        #     'mode': 'nearest'}
        #  },
        {'ResizeWithPadOrCropd': {
            'keys': ['image', 'label'],
            'spatial_size': def_spatial_size}
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
            'shear_range': radians(5),
            'translate_range': 0.05,
            'scale_range': 0.05,
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
        {'RandomGhosting': {
            'include': ['image'],
            'p': tiny_prob,
            'num_ghosts': (1, 4)
        }},
        {'RandomBlur': {
            'include': ['image', 'label'],
            'std': (0.1, 0.5),
            'p': low_prob}
        },
        {'RandomBiasField': {
            'include': ['image'],
            'p': high_prob,
            'coefficients': 0.5}
        },
        {'RandomMotion': {
            'include': ['image', 'label'],
            'p': low_prob,
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
        # {'Resized': {
        #     'keys': ['image', 'label'],
        #     'spatial_size': def_spatial_size,
        #     'mode': 'nearest'}
        #  },
        {'ResizeWithPadOrCropd': {
            'keys': ['image', 'label'],
            'spatial_size': def_spatial_size}
        },
        # 'ToTensord': {'keys': ['image', 'label']},

        # 'PrintDim': {'keys': ['image', 'label'], 'msg': 'after binarize and resize'},
        # 'AddChanneld': {'keys': ['image']},
        # 'SqueezeDimd': {'keys': ["image", "label"],
        #                 'dim': 0},
        {'NormalizeIntensityd': {'keys': ['image']}},
    ]
}

full_hyper_dict_cc = deepcopy(full_hyper_dict)
full_hyper_dict_cc['last_transform'].append({'CoordConvd': {'keys': ['image']}})

# new_full_dict_cc = deepcopy(full_hyper_dict_cc)
# new_full_dict_cc['monai_transform'].append({'ThreeDHaircutd': {
#             'keys': ['image', 'label'],
#             'prob': low_prob,
#             'index_range': 0.2}
#  })
# new_full_dict_cc['monai_transform'].append({'Anisotropiserd': {
#     'keys': ['image', 'label'],
#     'prob': low_prob,
#     'scale_range': (0.25, 0.8)}
#  })

minimal_hyper_dict = {
    'first_transform': [
        {'LoadImaged': {
            'keys': ['image', 'label']}},
        # {'ToTensord': {'keys': ['image', 'label']}},
        {'AddChanneld': {'keys': ['image', 'label']}},
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'BEFORE RESIZE'}},
        {'ResizeWithPadOrCropd': {
            'keys': ['image', 'label'],
            'spatial_size': def_spatial_size}
        },
        # {'ToTensord': {'keys': ['image', 'label']}},
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'AFTER RESIZE'}},
        # {'NormalizeIntensityd': {'keys': ['image']}},
        {'MyNormalizeIntensityd': {
            'keys': ['image'],
            'out_min_max': (-1, 1),
            # 'clamp_quantile': (.001, .999)
        }
        },
        # {'ToTensord': {'keys': ['image', 'label']}},
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'BEFORE Binarize'}},
        {'Binarized': {'keys': ['label'], 'lower_threshold': 0.5}},
    ],
    'last_transform': [
        # {'GaussianSmoothd': {
        #     'keys': ['label'],
        #     'sigma': .5}
        #  },

        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'BEFORE SECOND F_ING BINARIZE'}},
        {'Binarized': {
            'keys': ['label'],
            'lower_threshold': 0.25}
        },
        {'ToNumpyd': {'keys': ['image', 'label']}},
        {'ResizeWithPadOrCropd': {
            'keys': ['image', 'label'],
            'spatial_size': def_spatial_size}
        },
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'AFTER second binarized'}},
        # {'NormalizeIntensityd': {'keys': ['image']}},
        {'MyNormalizeIntensityd': {
            'keys': ['image'],
            'out_min_max': (-1, 1)}
        },
        {'ToTensord': {'keys': ['image', 'label']}},
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'THE END'}},
    ]
}

minimal_hyper_dict_cc = deepcopy(minimal_hyper_dict)
minimal_hyper_dict_cc['last_transform'].append({'CoordConvd': {'keys': ['image']}})

minimal_hyper_dict_altcc = deepcopy(minimal_hyper_dict)
minimal_hyper_dict_altcc['last_transform'].append({'CoordConvAltd': {'keys': ['image', 'label']}})

new_dict = {
    'first_transform': [
        {'LoadImaged': {'keys': ['image', 'label']}},
        {'AddChanneld': {'keys': ['image', 'label']}},
        {'ResizeWithPadOrCropd': {
            'keys': ['image', 'label'],
            'spatial_size': def_spatial_size}
        },
        {'NormalizeIntensityd': {'keys': ['image']}},
        {'Binarized': {'keys': ['label'], 'lower_threshold': 0.5}},
    ],
    # 'custom_transform': [
    # #     {'ThreeDHaircutd': {
    # #                 'keys': ['image', 'label'],
    # #                 'prob': low_prob,
    # #                 'index_range': 0.2}
    # #      },
    #     {'Anisotropiserd': {
    #         'keys': ['image', 'label'],
    #         'prob': low_prob,
    #         'scale_range': (0.25, 0.8)}
    #      }
    # ],
    # 'monai_transform': [
    #     # {'RandSpatialCropd': {'keys': ['image', 'label'],
    #     #                       'roi_size': min_small_crop_size,
    #     #                       'random_center': True,
    #     #                       'random_size': False}
    #     #  },
    #     {'RandHistogramShiftd': {
    #         'keys': ['image'],
    #         'num_control_points': (10, 15),
    #         'prob': low_prob}
    #      },
    #     # # TODO maybe 'Orientation': {} but it would interact with the flip,
    #     {'RandAffined': {
    #         'keys': ['image', 'label'],
    #         'prob': high_prob,
    #         'rotate_range': radians(5),  # 5 degrees converted in radians
    #         'shear_range': radians(5),
    #         'translate_range': 0.05,
    #         'scale_range': 0.05,
    #         'spatial_size': None,
    #         'padding_mode': 'border',
    #         'as_tensor_output': False}
    #      },
    #     {'Rand3DElasticd': {
    #         'keys': ['image', 'label'],
    #         'sigma_range': (1, 3),
    #         'magnitude_range': (3, 5),
    #         'prob': tiny_prob,
    #         'rotate_range': None,
    #         'shear_range': None,
    #         'translate_range': None,
    #         'scale_range': None,
    #         'spatial_size': None,
    #         'padding_mode': "reflection",
    #         # 'padding_mode': "border",
    #         # 'padding_mode': "zeros",
    #         'as_tensor_output': False}
    #      },
    #     {'ToTensord': {'keys': ['image', 'label']}},
    # ],
    # 'torchio_transform': [
    #     # {'RandomNoise': {
    #     #     'include': ['image'],
    #     #     'mean': 0,
    #     #     'std': (0.01, 0.1),
    #     #     'p': low_prob}
    #     #  },
    #     {'RandomGhosting': {
    #         'include': ['image'],
    #         'p': tiny_prob,
    #         'num_ghosts': (1, 4),
    #         'intensity': (0.3, 0.6)
    #     }},
    #     {'RandomBlur': {
    #         'include': ['image', 'label'],
    #         'std': (0.01, 0.07),
    #         'p': low_prob}
    #      },
    #     {'RandomBiasField': {
    #         'include': ['image'],
    #         'p': high_prob,
    #         'coefficients': 0.1}
    #      },
    #     # {'RandomMotion': {
    #     #     'include': ['image', 'label'],
    #     #     'p': low_prob,
    #     #     'num_transforms': 1}
    #     #  },
    #     {'ToTensord': {'keys': ['image', 'label']}},
    # ],
    'last_transform': [
        {'Binarized': {
            'keys': ['label'],
            'lower_threshold': 0.5}
        },
        {'ResizeWithPadOrCropd': {
            'keys': ['image', 'label'],
            'spatial_size': def_spatial_size}
        },
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'AFTER binarized'}},
        {'NormalizeIntensityd': {'keys': ['image']}},
    ]
}

new_dict_cc = deepcopy(new_dict)
new_dict_cc['last_transform'].append({'CoordConvd': {'keys': ['image']}})

curated_dict = {
    'first_transform': [
        {'LoadImaged': {
            'keys': ['image', 'label']}},
        # {'ToTensord': {'keys': ['image', 'label']}},
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'BEFORE RESIZE'}},
        {'AddChanneld': {'keys': ['image', 'label']}},
        {'ResizeWithPadOrCropd': {
            'keys': ['image', 'label'],
            'spatial_size': def_spatial_size}
        },
        # {'ToTensord': {'keys': ['image', 'label']}},
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'AFTER RESIZE'}},
        # {'NormalizeIntensityd': {'keys': ['image']}},
        {'MyNormalizeIntensityd': {
            'keys': ['image'],
            'out_min_max': (-1, 1),
            # 'clamp_quantile': (.001, .999)
        }
        },
        # {'ToTensord': {'keys': ['image', 'label']}},
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'BEFORE Binarize'}},
        {'Binarized': {'keys': ['label'], 'lower_threshold': 0.5}},
    ],
    # 'custom_transform': [
    # #     {'ThreeDHaircutd': {
    # #                 'keys': ['image', 'label'],
    # #                 'prob': low_prob,
    # #                 'index_range': 0.2}
    # #      },
    #     {'Anisotropiserd': {
    #         'keys': ['image', 'label'],
    #         'prob': low_prob,
    #         'scale_range': (0.25, 0.8)}
    #      }
    # ],
    'monai_transform': [
        # {'RandSpatialCropd': {'keys': ['image', 'label'],
        #                       'roi_size': min_small_crop_size,
        #                       'random_center': True,
        #                       'random_size': False}
        #  },
        {'RandHistogramShiftd': {
            'keys': ['image'],
            'num_control_points': (10, 15),
            'prob': low_prob}
        },
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'AFTER HIST SHIFT'}},
        # {'RandBiasFieldd': {
        #     'keys': ['image'],
        #     'coeff_range': (0, 0.05),
        #     'prob': high_prob}
        #  },
        # # TODO maybe 'Orientation': {} but it would interact with the flip,
        # {'ToTensord': {'keys': ['image', 'label']}},
        {'RandAffined': {
            'keys': ['image', 'label'],
            'prob': low_prob,
            'rotate_range': radians(5),
            'shear_range': radians(5),
            'translate_range': 0.05,
            'scale_range': 0.05,
            'padding_mode': 'border',
        }  # was False
        },
        {'Rand3DElasticd': {
            'keys': ['image', 'label'],
            'sigma_range': (1, 3),
            'magnitude_range': (3, 5),  # hyper_params['Rand3DElastic_magnitude_range']
            'prob': tiny_prob,
            'rotate_range': None,
            'shear_range': None,
            'translate_range': None,
            'scale_range': None,
            'padding_mode': "reflection",
            # 'padding_mode': "border",
            # 'padding_mode': "zeros",
        }
        },
    ],
    'torchio_transform': [
        {'ToTensord': {'keys': ['image', 'label']}},
        # {'RandomNoise': {
        #     'include': ['image'],
        #     'mean': 0,
        #     'std': (0.01, 0.1),
        #     'p': low_prob}
        #  },
        # {'RandomGhosting': {
        #     'include': ['image'],
        #     'p': tiny_prob,
        #     'num_ghosts': (1, 4),
        #     'intensity': (0.3, 0.6)
        # }},
        # {'RandomBlur': {
        #     'include': ['image', 'label'],
        #     'std': (0.01, 0.07),
        #     'p': low_prob}
        #  },
        {'RandomBiasField': {
            'include': ['image'],
            'p': low_prob,
            'coefficients': 0.1}
        },
        # {'RandomMotion': {
        #     'include': ['image', 'label'],
        #     'p': low_prob,
        #     'num_transforms': 1}
        #  },
        # {'ToNumpyd': {'keys': ['image', 'label']}},
    ],
    'last_transform': [
        # {'GaussianSmoothd': {
        #     'keys': ['label'],
        #     'sigma': .5}
        #  },

        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'BEFORE SECOND F_ING BINARIZE'}},
        {'Binarized': {
            'keys': ['label'],
            'lower_threshold': 0.25}
        },
        {'ToNumpyd': {'keys': ['image', 'label']}},
        {'ResizeWithPadOrCropd': {
            'keys': ['image', 'label'],
            'spatial_size': def_spatial_size}
        },
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'AFTER second binarized'}},
        # {'NormalizeIntensityd': {'keys': ['image']}},
        {'MyNormalizeIntensityd': {
            'keys': ['image'],
            'out_min_max': (-1, 1)}
        },
        {'ToTensord': {'keys': ['image', 'label']}},
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'THE END'}},
    ]
}

curated_dict_cc = deepcopy(curated_dict)
curated_dict_cc['last_transform'].append({'CoordConvd': {'keys': ['image']}})

std_dict = deepcopy(curated_dict)
for di in std_dict['first_transform']:
    for tr in di:
        if tr == 'MyNormalizeIntensityd':
            di[tr]['no_std'] = False
for di in std_dict['last_transform']:
    for tr in di:
        if tr == 'MyNormalizeIntensityd':
            di[tr]['no_std'] = False
std_dict_cc = deepcopy(std_dict)
std_dict_cc['last_transform'].append({'CoordConvd': {'keys': ['image']}})

mod_full_dict = deepcopy(full_hyper_dict)
mod_full_dict['last_transform'] = curated_dict['last_transform']
mod_full_dict['first_transform'] = curated_dict['first_transform']
mod_full_dict_cc = deepcopy(mod_full_dict)
mod_full_dict_cc['last_transform'].append({'CoordConvd': {'keys': ['image']}})

test_dict = {
    'first_transform': [
        {'LoadImaged': {
            'keys': ['image', 'label']}},
        {'ToTensord': {'keys': ['image', 'label']}},
        {'AddChanneld': {'keys': ['image', 'label']}},
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'BEFORE RESIZE'}},
        {'ResizeWithPadOrCropd': {
            'keys': ['image', 'label'],
            'spatial_size': def_spatial_size}
        },
        {'ToTensord': {'keys': ['image', 'label']}},
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'AFTER RESIZE'}},
        {'PrintDim': {'keys': ['image', 'label'], 'msg': 'BEFORE NORMALIZE start time'}},
        {'MyNormalizeIntensityd': {'keys': ['image']}},
        {'PrintDim': {'keys': ['image', 'label'], 'msg': 'AFTER NORMALIZE end time'}},
        {'ToTensord': {'keys': ['image', 'label']}},
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'BEFORE Binarize'}},
        {'Binarized': {'keys': ['label'], 'lower_threshold': 0.5}},
    ],
    'monai_transform': [
        # # TODO maybe 'Orientation': {} but it would interact with the flip,
        {'RandAffined': {
            'keys': ['image', 'label'],
            'prob': 1,
            'rotate_range': radians(5),
            'shear_range': radians(5),
            'translate_range': 0.05,
            'scale_range': 0.05,
            'padding_mode': 'border',
            'as_tensor_output': True}  # was False
        },
        {'Rand3DElasticd': {
            'keys': ['image', 'label'],
            'sigma_range': (1, 3),
            'magnitude_range': (3, 5),  # hyper_params['Rand3DElastic_magnitude_range']
            'prob': 1,
            'rotate_range': None,
            'shear_range': None,
            'translate_range': None,
            'scale_range': None,
            'padding_mode': "reflection",
            # 'padding_mode': "border",
            # 'padding_mode': "zeros",
            'as_tensor_output': True}
        },
    ],
    'torchio_transform': [
        {'ToTensord': {'keys': ['image', 'label']}},
        {'RandomBiasField': {
            'include': ['image'],
            'p': 1,
            'coefficients': 0.1}
        },
    ],
    'last_transform': [
        {'Binarized': {
            'keys': ['label'],
            'lower_threshold': 0.25}
        },
        {'ResizeWithPadOrCropd': {
            'keys': ['image', 'label'],
            'spatial_size': def_spatial_size}
        },
        {'MyNormalizeIntensityd': {'keys': ['image']}},
    ]
}

crop_test = {
    'first_transform': [
        {'LoadImaged': {
            'keys': ['image', 'label']}},
        # {'ToTensord': {'keys': ['image', 'label']}},
        {'AddChanneld': {'keys': ['image', 'label']}},
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'BEFORE RESIZE'}},
        # {'ResizeWithPadOrCropd': {
        #     'keys': ['image', 'label'],
        #     'spatial_size': def_spatial_size}
        #  },
        # {'ToTensord': {'keys': ['image', 'label']}},
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'AFTER RESIZE'}},
        # {'NormalizeIntensityd': {'keys': ['image']}},
        {'MyNormalizeIntensityd': {
            'keys': ['image'],
            'out_min_max': (-1, 1),
            # 'clamp_quantile': (.001, .999)
        }
        },
        # {'ToTensord': {'keys': ['image', 'label']}},
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'BEFORE Binarize'}},
        {'Binarized': {'keys': ['label'], 'lower_threshold': 0.5}},
    ],
    # 'custom_transform': [
    # #     {'ThreeDHaircutd': {
    # #                 'keys': ['image', 'label'],
    # #                 'prob': low_prob,
    # #                 'index_range': 0.2}
    # #      },
    #     {'Anisotropiserd': {
    #         'keys': ['image', 'label'],
    #         'prob': low_prob,
    #         'scale_range': (0.25, 0.8)}
    #      }
    # ],
    'monai_transform': [
        # {'RandSpatialCropd': {'keys': ['image', 'label'],
        #                       'roi_size': min_small_crop_size,
        #                       'random_center': True,
        #                       'random_size': False}
        #  },
        {'RandHistogramShiftd': {
            'keys': ['image'],
            'num_control_points': (10, 15),
            'prob': low_prob}
        },
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'AFTER HIST SHIFT'}},
        # {'RandBiasFieldd': {
        #     'keys': ['image'],
        #     'coeff_range': (0, 0.05),
        #     'prob': high_prob}
        #  },
        # # TODO maybe 'Orientation': {} but it would interact with the flip,
        {'RandAffined': {
            'keys': ['image', 'label'],
            'prob': low_prob,
            'rotate_range': radians(5),
            'shear_range': radians(5),
            'translate_range': 0.05,
            'scale_range': 0.05,
            'padding_mode': 'border',
            'as_tensor_output': True}  # was False
        },
        {'Rand3DElasticd': {
            'keys': ['image', 'label'],
            'sigma_range': (1, 3),
            'magnitude_range': (3, 5),  # hyper_params['Rand3DElastic_magnitude_range']
            'prob': tiny_prob,
            'rotate_range': None,
            'shear_range': None,
            'translate_range': None,
            'scale_range': None,
            'padding_mode': "reflection",
            # 'padding_mode': "border",
            # 'padding_mode': "zeros",
            'as_tensor_output': True}
        },
    ],
    'torchio_transform': [
        {'ToTensord': {'keys': ['image', 'label']}},
        # {'RandomNoise': {
        #     'include': ['image'],
        #     'mean': 0,
        #     'std': (0.01, 0.1),
        #     'p': low_prob}
        #  },
        # {'RandomGhosting': {
        #     'include': ['image'],
        #     'p': tiny_prob,
        #     'num_ghosts': (1, 4),
        #     'intensity': (0.3, 0.6)
        # }},
        # {'RandomBlur': {
        #     'include': ['image', 'label'],
        #     'std': (0.01, 0.07),
        #     'p': low_prob}
        #  },
        {'RandomBiasField': {
            'include': ['image'],
            'p': low_prob,
            'coefficients': 0.1}
        },
        # {'RandomMotion': {
        #     'include': ['image', 'label'],
        #     'p': low_prob,
        #     'num_transforms': 1}
        #  },
        # {'ToNumpyd': {'keys': ['image', 'label']}},
    ],
    'last_transform': [
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'end of augmentations'}},
        # {'GaussianSmoothd': {
        #     'keys': ['label'],
        #     'sigma': .5}
        #  },

        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'BEFORE SECOND F_ING BINARIZE'}},
        {'Binarized': {
            'keys': ['label'],
            'lower_threshold': 0.25}
        },
        # {'ResizeWithPadOrCropd': {
        #     'keys': ['image', 'label'],
        #     'spatial_size': def_spatial_size}
        #  },
        # {'ToTensord': {'keys': ['image', 'label']}},
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'AFTER second binarized'}},
        # {'NormalizeIntensityd': {'keys': ['image']}},
        {'MyNormalizeIntensityd': {
            'keys': ['image'],
            'out_min_max': (-1, 1)}
        },
        # {'ToNumpyd': {'keys': ['image', 'label']}},
        {'ToTensord': {'keys': ['image', 'label']}},
    ],
    'crop': [
        {'RandCropByPosNegLabeld': {
            'keys': ['image', 'label'],
            'label_key': 'label',
            # UNet cannot take every dimension min_small_crop_size (91, 121, 91) does not work for example
            'spatial_size': [80, 96, 80],
            # 'spatial_size': [32, 32, 32],
            # 'spatial_size': min_small_crop_size,
            'pos': 1,
            'neg': 1,
            'num_samples': 4}},
    ]
}

crop_test_cc = deepcopy(crop_test)
crop_test_cc['last_transform'].append({'CoordConvd': {'keys': ['image']}})

unetr_dict = {
    'first_transform': [
        {'LoadImaged': {
            'keys': ['image', 'label']}},
        # {'ToTensord': {'keys': ['image', 'label']}},
        {'AddChanneld': {'keys': ['image', 'label']}},
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'BEFORE RESIZE'}},
        # {'ResizeWithPadOrCropd': {
        #     'keys': ['image', 'label'],
        #     'spatial_size': def_spatial_size}
        #  },
        # {'ToTensord': {'keys': ['image', 'label']}},
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'AFTER RESIZE'}},
        # {'NormalizeIntensityd': {'keys': ['image']}},
        {'MyNormalizeIntensityd': {
            'keys': ['image'],
            'out_min_max': (0, 1),
            # 'clamp_quantile': (.001, .999)
        }
        },
        # {'ToTensord': {'keys': ['image', 'label']}},
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'BEFORE Binarize'}},
        {'Binarized': {'keys': ['label'], 'lower_threshold': 0.5}},
    ],
    # 'custom_transform': [
    # #     {'ThreeDHaircutd': {
    # #                 'keys': ['image', 'label'],
    # #                 'prob': low_prob,
    # #                 'index_range': 0.2}
    # #      },
    #     {'Anisotropiserd': {
    #         'keys': ['image', 'label'],
    #         'prob': low_prob,
    #         'scale_range': (0.25, 0.8)}
    #      }
    # ],
    'monai_transform': [
        # {'RandSpatialCropd': {'keys': ['image', 'label'],
        #                       'roi_size': min_small_crop_size,
        #                       'random_center': True,
        #                       'random_size': False}
        #  },
        {'RandHistogramShiftd': {
            'keys': ['image'],
            'num_control_points': (10, 15),
            'prob': low_prob}
        },
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'AFTER HIST SHIFT'}},
        # {'RandBiasFieldd': {
        #     'keys': ['image'],
        #     'coeff_range': (0, 0.05),
        #     'prob': high_prob}
        #  },
        # # TODO maybe 'Orientation': {} but it would interact with the flip,
        {'RandAffined': {
            'keys': ['image', 'label'],
            'prob': low_prob,
            'rotate_range': radians(5),
            'shear_range': radians(5),
            'translate_range': 0.05,
            'scale_range': 0.05,
            'padding_mode': 'border',
            'as_tensor_output': True}  # was False
        },
        {'Rand3DElasticd': {
            'keys': ['image', 'label'],
            'sigma_range': (1, 3),
            'magnitude_range': (3, 5),  # hyper_params['Rand3DElastic_magnitude_range']
            'prob': tiny_prob,
            'rotate_range': None,
            'shear_range': None,
            'translate_range': None,
            'scale_range': None,
            'padding_mode': "reflection",
            # 'padding_mode': "border",
            # 'padding_mode': "zeros",
            'as_tensor_output': True}
        },
    ],
    'torchio_transform': [
        {'ToTensord': {'keys': ['image', 'label']}},
        # {'RandomNoise': {
        #     'include': ['image'],
        #     'mean': 0,
        #     'std': (0.01, 0.1),
        #     'p': low_prob}
        #  },
        # {'RandomGhosting': {
        #     'include': ['image'],
        #     'p': tiny_prob,
        #     'num_ghosts': (1, 4),
        #     'intensity': (0.3, 0.6)
        # }},
        # {'RandomBlur': {
        #     'include': ['image', 'label'],
        #     'std': (0.01, 0.07),
        #     'p': low_prob}
        #  },
        {'RandomBiasField': {
            'include': ['image'],
            'p': low_prob,
            'coefficients': 0.1}
        },
        # {'RandomMotion': {
        #     'include': ['image', 'label'],
        #     'p': low_prob,
        #     'num_transforms': 1}
        #  },
        # {'ToNumpyd': {'keys': ['image', 'label']}},
    ],
    'last_transform': [
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'end of augmentations'}},
        # {'GaussianSmoothd': {
        #     'keys': ['label'],
        #     'sigma': .5}
        #  },

        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'BEFORE SECOND F_ING BINARIZE'}},
        {'Binarized': {
            'keys': ['label'],
            'lower_threshold': 0.25}
        },
        # {'ResizeWithPadOrCropd': {
        #     'keys': ['image', 'label'],
        #     'spatial_size': def_spatial_size}
        #  },
        # {'ToTensord': {'keys': ['image', 'label']}},
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'AFTER second binarized'}},
        # {'NormalizeIntensityd': {'keys': ['image']}},
        {'MyNormalizeIntensityd': {
            'keys': ['image'],
            'out_min_max': (0, 1)}
        },
        # {'ToNumpyd': {'keys': ['image', 'label']}},
        {'ToTensord': {'keys': ['image', 'label']}},
    ],
    'crop': [
        {'RandCropByPosNegLabeld': {
            'keys': ['image', 'label'],
            'label_key': 'label',
            'spatial_size': [80, 80, 80],
            'pos': 1,
            'neg': 1,
            'num_samples': 4}},
    ],
    'unetr_transform': [
        {'RandFlipd': {
            'keys': ["image", "label"],
            'spatial_axis': [0],
            'prob': low_prob}
        },
        {'RandFlipd': {
            'keys': ["image", "label"],
            'spatial_axis': [1],
            'prob': low_prob}
        },
        {'RandFlipd': {
            'keys': ["image", "label"],
            'spatial_axis': [2],
            'prob': low_prob}
        },
        {'RandRotate90d': {
            'keys': ["image", "label"],
            'prob': low_prob,
            'max_k': 3}
        },
        {'RandShiftIntensityd': {
            'keys': ["image"],
            'offsets': 0.10,
            'prob': high_prob}
        },
    ],
}

unetr_dict_cc = deepcopy(unetr_dict)
unetr_dict_cc['last_transform'].append({'CoordConvd': {'keys': ['image']}})

unetr_dict_lastflip = {
    'first_transform': [
        {'LoadImaged': {
            'keys': ['image', 'label']}},
        # {'ToTensord': {'keys': ['image', 'label']}},
        {'AddChanneld': {'keys': ['image', 'label']}},
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'BEFORE RESIZE'}},
        # {'ResizeWithPadOrCropd': {
        #     'keys': ['image', 'label'],
        #     'spatial_size': def_spatial_size}
        #  },
        # {'ToTensord': {'keys': ['image', 'label']}},
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'AFTER RESIZE'}},
        # {'NormalizeIntensityd': {'keys': ['image']}},
        {'MyNormalizeIntensityd': {
            'keys': ['image'],
            'out_min_max': (0, 1),
            # 'clamp_quantile': (.001, .999)
        }
        },
        # {'ToTensord': {'keys': ['image', 'label']}},
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'BEFORE Binarize'}},
        {'Binarized': {'keys': ['label'], 'lower_threshold': 0.5}},
    ],
    # 'custom_transform': [
    # #     {'ThreeDHaircutd': {
    # #                 'keys': ['image', 'label'],
    # #                 'prob': low_prob,
    # #                 'index_range': 0.2}
    # #      },
    #     {'Anisotropiserd': {
    #         'keys': ['image', 'label'],
    #         'prob': low_prob,
    #         'scale_range': (0.25, 0.8)}
    #      }
    # ],
    'monai_transform': [
        # {'RandSpatialCropd': {'keys': ['image', 'label'],
        #                       'roi_size': min_small_crop_size,
        #                       'random_center': True,
        #                       'random_size': False}
        #  },
        {'RandHistogramShiftd': {
            'keys': ['image'],
            'num_control_points': (10, 15),
            'prob': low_prob}
        },
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'AFTER HIST SHIFT'}},
        # {'RandBiasFieldd': {
        #     'keys': ['image'],
        #     'coeff_range': (0, 0.05),
        #     'prob': high_prob}
        #  },
        # # TODO maybe 'Orientation': {} but it would interact with the flip,
        {'RandAffined': {
            'keys': ['image', 'label'],
            'prob': low_prob,
            'rotate_range': radians(5),
            'shear_range': radians(5),
            'translate_range': 0.05,
            'scale_range': 0.05,
            'padding_mode': 'border',
            'as_tensor_output': True}  # was False
        },
        {'Rand3DElasticd': {
            'keys': ['image', 'label'],
            'sigma_range': (1, 3),
            'magnitude_range': (3, 5),  # hyper_params['Rand3DElastic_magnitude_range']
            'prob': tiny_prob,
            'rotate_range': None,
            'shear_range': None,
            'translate_range': None,
            'scale_range': None,
            'padding_mode': "reflection",
            # 'padding_mode': "border",
            # 'padding_mode': "zeros",
            'as_tensor_output': True}
        },
    ],
    'torchio_transform': [
        {'ToTensord': {'keys': ['image', 'label']}},
        # {'RandomNoise': {
        #     'include': ['image'],
        #     'mean': 0,
        #     'std': (0.01, 0.1),
        #     'p': low_prob}
        #  },
        # {'RandomGhosting': {
        #     'include': ['image'],
        #     'p': tiny_prob,
        #     'num_ghosts': (1, 4),
        #     'intensity': (0.3, 0.6)
        # }},
        # {'RandomBlur': {
        #     'include': ['image', 'label'],
        #     'std': (0.01, 0.07),
        #     'p': low_prob}
        #  },
        {'RandomBiasField': {
            'include': ['image'],
            'p': low_prob,
            'coefficients': 0.1}
        },
        # {'RandomMotion': {
        #     'include': ['image', 'label'],
        #     'p': low_prob,
        #     'num_transforms': 1}
        #  },
        # {'ToNumpyd': {'keys': ['image', 'label']}},
    ],
    'unetr_transform': [
        {'RandFlipd': {
            'keys': ["image", "label"],
            'spatial_axis': [0],
            'prob': low_prob}
        },
        # {'RandFlipd': {
        #     'keys': ["image", "label"],
        #     'spatial_axis': [1],
        #     'prob': low_prob}
        #  },
        # {'RandFlipd': {
        #     'keys': ["image", "label"],
        #     'spatial_axis': [2],
        #     'prob': low_prob}
        #  },
        # {'RandRotate90d': {
        #     'keys': ["image", "label"],
        #     'prob': low_prob,
        #     'max_k': 3}
        #  },
        {'RandShiftIntensityd': {
            'keys': ["image"],
            'offsets': 0.10,
            'prob': high_prob}
         },
    ],
    'last_transform': [
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'end of augmentations'}},
        # {'GaussianSmoothd': {
        #     'keys': ['label'],
        #     'sigma': .5}
        #  },

        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'BEFORE SECOND F_ING BINARIZE'}},
        {'Binarized': {
            'keys': ['label'],
            'lower_threshold': 0.25}
        },
        # {'ResizeWithPadOrCropd': {
        #     'keys': ['image', 'label'],
        #     'spatial_size': def_spatial_size}
        #  },
        # {'ToTensord': {'keys': ['image', 'label']}},
        # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'AFTER second binarized'}},
        # {'NormalizeIntensityd': {'keys': ['image']}},
        {'MyNormalizeIntensityd': {
            'keys': ['image'],
            'out_min_max': (0, 1)}
        },
        # {'ToNumpyd': {'keys': ['image', 'label']}},
        {'ToTensord': {'keys': ['image', 'label']}},
    ],
    'crop': [
        {'RandCropByPosNegLabeld': {
            'keys': ['image', 'label'],
            'label_key': 'label',
            'spatial_size': [80, 80, 80],
            'pos': 1,
            'neg': 1,
            'num_samples': 4}},
    ]
}

shear_min = 0.05
shear_max = 0.1
# low_prob = high_prob = tiny_prob = 1
unetr_cc = {
    'first_transform': [
        {'LoadImaged': {
            'keys': ['image', 'label']}},
        {'EnsureChannelFirstd': {'keys': ['image', 'label']}},
        {'ResizeWithPadOrCropd': {
            'keys': ['image', 'label'],
            'spatial_size': def_spatial_size}
         },
        # {'Resized': {
        #     'keys': ['image', 'label'],
        #     'spatial_size': def_spatial_size}
        #  },
        {'MyNormalizeIntensityd': {
            'keys': ['image'],
            'out_min_max': (0, 1),
            # 'clamp_quantile': (.001, .999)
        }
        },
        {'Binarized': {'keys': ['label'], 'lower_threshold': 0.5}},
    ],
    'monai_transform': [
        {'RandHistogramShiftd': {
            'keys': ['image'],
            'num_control_points': 20,
            'prob': low_prob}
         },
        # TODO maybe 'Orientation': {} but it would interact with the flip,
        # {'RandAffined': {
        #     'keys': ['image', 'label'],
        #     'prob': low_prob,
        #     'rotate_range': radians(5),
        #     'shear_range': radians(5),
        #     'translate_range': 0.05,
        #     'scale_range': 0.05,
        #     'padding_mode': 'border',
        #     'mode': 'nearest'}  # was False
        #  },
        {'Rand3DElasticd': {
            'keys': ['image', 'label'],
            'sigma_range': (3, 15),
            'magnitude_range': (3, 10),  # hyper_params['Rand3DElastic_magnitude_range']
            'prob': tiny_prob,
            'rotate_range': (radians(1), radians(10)),
            'shear_range': ([(shear_min, shear_max) for i in range(6)]),
            'translate_range': (0.5, 3),
            'scale_range': (0.02, 0.15),
            'padding_mode': "reflection",
            'mode': 'nearest',
            # 'padding_mode': "border",
            # 'padding_mode': "zeros"
        }
        },
        {'RandGibbsNoised': {'keys': ['image'],
                             'prob': high_prob,
                             'alpha': (0.5, 0.7)
                             },
         },
        {'RandRicianNoised': {'keys': ['image'],
                              'prob': low_prob,
                              'mean': 0.1,
                              'std': 0.025
                              },
         },
        {'RandKSpaceSpikeNoised': {'keys': ['image'],
                                   'prob': low_prob,
                                   'intensity_range': (8, 10),
                                   },
         },
        {'RandBiasFieldd': {
            'keys': ['image'],
            'prob': high_prob,
            'coeff_range': (0.0, 0.1)}
         },
    ],
    # 'torchio_transform': [
    #     # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'PrintDim before ToTensord'}},
    #     # {'ToTensord': {'keys': ['image', 'label']}},
    #     {'RandomBiasField': {
    #         'include': ['image'],
    #         'p': low_prob,
    #         'coefficients': 0.05}
    #      },
    #     # {'RandomNoise': {
    #     #     'include': ['image'],
    #     #     'p': low_prob,
    #     #     'mean': 0.5,
    #     #     'std': (0, 0.025)}
    #     # },
    # ],
    'unetr_transform': [
        {'RandFlipd': {
            'keys': ["image", "label"],
            'spatial_axis': [0],
            'prob': low_prob}
         },
        {'RandShiftIntensityd': {
            'keys': ["image"],
            'offsets': 0.10,
            'prob': high_prob}
         },
    ],
    'last_transform': [
        {'Binarized': {
            'keys': ['label'],
            'lower_threshold': 0.25}
         },
        {'MyNormalizeIntensityd': {
            'keys': ['image'],
            'out_min_max': (0, 1)}
         },
        # {'ToTensord': {'keys': ['image', 'label']}},
        {'CoordConvd': {'keys': ['image']}}
    ],
}

unetr_cc_std_norm = deepcopy(unetr_cc)
unetr_cc_std_norm['first_transform'][3]['MyNormalizeIntensityd']['no_std'] = False
unetr_cc_std_norm['last_transform'][1]['MyNormalizeIntensityd']['no_std'] = False


unetr_cc_resize = deepcopy(unetr_cc)
unetr_cc_resize['first_transform'][2] = {'Resized': {
            'keys': ['image', 'label'],
            'spatial_size': def_spatial_size}
         }


unetr_cc_patches = deepcopy(unetr_cc)
del unetr_cc_patches['first_transform'][2]
unetr_cc_patches['patches'] = [
    {'RandSpatialCropSamplesd': {
        'keys': ['image', 'label'],
        'roi_size': [64, 64, 64],
        'num_samples': 2,
        'random_center': True,
        'random_size': False}},
    # {'RandCropByPosNegLabeld': {
    #     'keys': ['image', 'label'],
    #     'label_key': 'label',
    #     'spatial_size': [80, 80, 80],
    #     'pos': 1,
    #     'neg': 1,
    #     'num_samples': 4}},
    # {'RandFlipd': {
    #     'keys': ["image", "label"],
    #     'spatial_axis': [1],
    #     'prob': low_prob}
    #  },
    # {'RandFlipd': {
    #     'keys': ["image", "label"],
    #     'spatial_axis': [2],
    #     'prob': low_prob}
    #  },
]


swinunetr_cc_patches = deepcopy(unetr_cc_patches)
# SwinUNETR requires an images size divisible by 12 and divisible by 2 five times ... So, multiples of 32 essentially
swinunetr_cc_patches['patches'][0]['RandSpatialCropSamplesd']['roi_size'] = [64, 64, 64]

low_prob = high_prob = tiny_prob = 1
unetr_aug_test = {
    'first_transform': [
        {'LoadImaged': {
            'keys': ['image', 'label']}},
        {'EnsureChannelFirstd': {'keys': ['image', 'label']}},
        {'ResizeWithPadOrCropd': {
            'keys': ['image', 'label'],
            'spatial_size': def_spatial_size}
         },
        {'MyNormalizeIntensityd': {
            'keys': ['image'],
            'out_min_max': (0, 1),
            # 'clamp_quantile': (.001, .999)
        }
        },
        {'Binarized': {'keys': ['label'], 'lower_threshold': 0.5}},
    ],
    'monai_transform': [
        {'RandHistogramShiftd': {
            'keys': ['image'],
            'num_control_points': 20,
            'prob': low_prob}
         },
        # TODO maybe 'Orientation': {} but it would interact with the flip,
        {'RandAffined': {
            'keys': ['image', 'label'],
            'prob': low_prob,
            'rotate_range': radians(5),
            'shear_range': radians(5),
            'translate_range': 0.05,
            'scale_range': 0.05,
            'padding_mode': 'border',
            'mode': 'nearest'}  # was False
         },
        {'Rand3DElasticd': {
            'keys': ['image', 'label'],
            'sigma_range': (1, 3),
            'magnitude_range': (3, 5),  # hyper_params['Rand3DElastic_magnitude_range']
            'prob': tiny_prob,
            'rotate_range': None,
            'shear_range': None,
            'translate_range': None,
            'scale_range': None,
            'padding_mode': "reflection",
            'mode': 'nearest',
            # 'padding_mode': "border",
            # 'padding_mode': "zeros"
        }
        },
        {'RandGibbsNoised': {'keys': ['image'],
                             'prob': high_prob,
                             'alpha': (0.5, 0.7)
                             },
         },
        {'RandRicianNoised': {'keys': ['image'],
                              'prob': 1,
                              'mean': 0.1,
                              'std': 0.025
                              },
         },
        {'RandKSpaceSpikeNoised': {'keys': ['image'],
                                   'prob': high_prob,
                                   'intensity_range': (8, 10),
                                   },
         },
         {'RandBiasFieldd': {
            'keys': ['image'],
            'prob': low_prob,
            'coeff_range': (0.1, 0.1)}
         },
    ],
    # 'torchio_transform': [
    #     # {'PrintDim': {'keys': ['image', 'label'], 'msg': 'PrintDim before ToTensord'}},
    #     {'ToTensord': {'keys': ['image', 'label']}},
    #     {'RandomBiasField': {
    #         'include': ['image'],
    #         'p': low_prob,
    #         'coefficients': 0.05}
    #      },
    #     # {'RandomNoise': {
    #     #     'include': ['image'],
    #     #     'p': low_prob,
    #     #     'mean': 0.5,
    #     #     'std': (0, 0.025)}
    #     # },
    # ],
    'unetr_transform': [
        # {'RandFlipd': {
        #     'keys': ["image", "label"],
        #     'spatial_axis': [0],
        #     'prob': low_prob}
        #  },
        {'RandShiftIntensityd': {
            'keys': ["image"],
            'offsets': [0.10, 0.10],
            'prob': high_prob}
         },
    ],
    'last_transform': [
        {'Binarized': {
            'keys': ['label'],
            'lower_threshold': 0.25}
         },
        {'MyNormalizeIntensityd': {
            'keys': ['image'],
            'out_min_max': (0, 1)}
         },
        {'ToTensord': {'keys': ['image', 'label']}},
        {'CoordConvd': {'keys': ['image']}}
    ],
}

unetr_no_aug = {
    'first_transform': [
        {'LoadImaged': {
            'keys': ['image', 'label']}},
        {'EnsureChannelFirstd': {'keys': ['image', 'label']}},
        {'ResizeWithPadOrCropd': {
            'keys': ['image', 'label'],
            'spatial_size': def_spatial_size}
         },
        {'MyNormalizeIntensityd': {
            'keys': ['image'],
            'out_min_max': (0, 1),
            # 'clamp_quantile': (.001, .999)
        }
        },
        {'Binarized': {'keys': ['label'], 'lower_threshold': 0.5}},
    ],
    'mid_transform': [

    ],
    'last_transform': [
        {'Binarized': {
            'keys': ['label'],
            'lower_threshold': 0.25}
         },
        {'MyNormalizeIntensityd': {
            'keys': ['image'],
            'out_min_max': (0, 1)}
         },
        {'CoordConvd': {'keys': ['image']}}
    ],
}

shear_min = 0.05
shear_max = 0.1
unetr_elastic = deepcopy(unetr_no_aug)
unetr_elastic['mid_transform'].append({'Rand3DElasticd': {
            'keys': ['image', 'label'],
            'sigma_range': (10, 10),
            'magnitude_range': (5, 5),  # hyper_params['Rand3DElastic_magnitude_range']
            'prob': tiny_prob,
            # 'rotate_range': (radians(10), radians(10)),
            # 'shear_range': ([(shear_min, shear_max) for i in range(6)]),
            # 'translate_range': (3, 3),
            # 'scale_range': (0.1, 0.1),
            'padding_mode': "zeros",
            'mode': 'nearest',
            # 'padding_mode': "border",
            # 'padding_mode': "zeros"
        }
        })

unetr_gibbs = deepcopy(unetr_no_aug)
unetr_gibbs['mid_transform'].append({
    'RandGibbsNoised': {'keys': ['image'],
                        'prob': high_prob,
                        'alpha': (0.4, 0.6)
                        },
})

unetr_spike = deepcopy(unetr_no_aug)
unetr_spike['mid_transform'].append({
    'RandKSpaceSpikeNoised': {'keys': ['image'],
                              'prob': high_prob,
                              'intensity_range': (8, 10)
                              },
})

unetr_rician = deepcopy(unetr_no_aug)
unetr_rician['mid_transform'].append({'RandRicianNoised': {'keys': ['image'],
                                                           'prob': 1,
                                                           'mean': 0.5,
                                                           'std': 0.025,
                                                           },
})

unetr_hist = deepcopy(unetr_no_aug)
unetr_hist['mid_transform'].append(
    {'RandHistogramShiftd': {
            'keys': ['image'],
            'num_control_points': 20,
            'prob': low_prob}
     },
)

unetr_bias = deepcopy(unetr_no_aug)
unetr_bias['mid_transform'].append(
    {'RandBiasFieldd': {
            'keys': ['image'],
            'prob': low_prob,
            'coeff_range': (0.1, 0.1)}
     },
    # {'RandomBiasField': {
    #         'include': ['image'],
    #         'p': low_prob,
    #         'coefficients': (0.1, 0.1)}
    # },
)

unetr_shift = deepcopy(unetr_no_aug)
unetr_shift['mid_transform'].append(
    {'RandShiftIntensityd': {
            'keys': ["image"],
            'offsets': 0.10,
            'prob': high_prob}
     },
)


def destructive_all_noises(noise_value):
    # noise_value is a percentage of the max value of the transformations (e.g. 10 for 0.1)
    noise_value = int(noise_value) / 100
    # adding transformation that will be kept during segmentation (add it to last_transform)
    tr_dict = deepcopy(unetr_cc)
    # The image is extremely blurry and could hardly be interpreted by a human
    gibbs_max_value = 0.425
    rician_max_value = 0.5
    rician_std_param = (rician_max_value * noise_value,)
    bias_field_max_value = 0.1
    to_add_list = [
        {'GibbsNoised': {'keys': ['image'],
                         'alpha': gibbs_max_value * noise_value
                         },
         },
        {'RandRicianNoised': {'keys': ['image'],
                              'prob': 1,
                              'mean': 0.1,
                              'std': rician_std_param,
                              'relative': True,
                              'sample_std': False,
                              'channel_wise': True,
                              },
         },
        {'RandBiasFieldd': {
            'keys': ['image'],
            'prob': 1,
            'coeff_range': (bias_field_max_value * noise_value, bias_field_max_value * noise_value),
            },
         },
    ]
    # prepend to last_transform
    tr_dict['last_transform'] = to_add_list + tr_dict['last_transform']
    return tr_dict


def destructive_gibbs(noise_value):
    # noise_value is a percentage of the max value of the transformations (e.g. 10 for 0.1)
    noise_value = int(noise_value) / 100
    # adding transformation that will be kept during segmentation (add it to last_transform)
    tr_dict = deepcopy(unetr_cc)
    # The image is extremely blurry and could hardly be interpreted by a human
    gibbs_max_value = 0.85
    to_add_list = [
        {'GibbsNoised': {'keys': ['image'],
                         'alpha': gibbs_max_value * noise_value
                         },
         },
    ]
    # prepend to last_transform
    tr_dict['last_transform'] = to_add_list + tr_dict['last_transform']
    return tr_dict


def destructive_rician(noise_value):
    # noise_value is a percentage of the max value of the transformations (e.g. 10 for 0.1)
    noise_value = int(noise_value) / 100
    # adding transformation that will be kept during segmentation (add it to last_transform)
    tr_dict = deepcopy(unetr_cc)
    # The image is extremely blurry and could hardly be interpreted by a human
    rician_max_value = 1
    rician_std_param = (rician_max_value * noise_value,)
    to_add_list = [
        {'RandRicianNoised': {'keys': ['image'],
                              'prob': 1,
                              'mean': 0.1,
                              'std': rician_std_param,
                              'relative': True,
                              'sample_std': False,
                              'channel_wise': True,
                              },
         },
    ]
    # prepend to last_transform
    tr_dict['last_transform'] = to_add_list + tr_dict['last_transform']
    return tr_dict


def destructive_bias(noise_value):
    # noise_value is a percentage of the max value of the transformations (e.g. 10 for 0.1)
    noise_value = int(noise_value) / 100
    # adding transformation that will be kept during segmentation (add it to last_transform)
    tr_dict = deepcopy(unetr_cc)
    # The image is extremely blurry and could hardly be interpreted by a human
    bias_field_max_value = 0.2
    to_add_list = [
        {'RandBiasFieldd': {
            'keys': ['image'],
            'prob': 1,
            'coeff_range': (bias_field_max_value * noise_value, bias_field_max_value * noise_value),
            },
         },
    ]
    # prepend to last_transform
    tr_dict['last_transform'] = to_add_list + tr_dict['last_transform']
    return tr_dict
