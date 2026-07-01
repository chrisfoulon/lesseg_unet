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
        {'EnsureChannelFirstd': {'keys': ['image', 'label']}},
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
        {'EnsureChannelFirstd': {'keys': ['image', 'label']}},
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
        {'EnsureChannelFirstd': {'keys': ['image', 'label']}},
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
        {'EnsureChannelFirstd': {'keys': ['image', 'label']}},
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
        {'EnsureChannelFirstd': {'keys': ['image', 'label']}},
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
        {'EnsureChannelFirstd': {'keys': ['image', 'label']}},
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
        {'EnsureChannelFirstd': {'keys': ['image', 'label']}},
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
        {'EnsureChannelFirstd': {'keys': ['image', 'label']}},
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
            'shear_range': ([(shear_min, shear_max) for _ in range(6)]),
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
    # CRITICAL for lesion segmentation: use RandCropByPosNegLabeld
    # This ensures patches are centered on lesions (pos) and healthy tissue (neg)
    {'RandCropByPosNegLabeld': {
        'keys': ['image', 'label'],
        'label_key': 'label',
        'spatial_size': [96, 96, 96],  # Larger patches for better context
        'pos': 2,  # 2 patches centered on lesion voxels
        'neg': 1,  # 1 patch on healthy tissue
        'num_samples': 3}},  # Total 3 patches per image per iteration
]


swinunetr_cc_patches = deepcopy(unetr_cc_patches)
# SwinUNETR requires image size divisible by 32
# unetr_cc_patches already uses 96x96x96 patches, which is perfect for SwinUNETR
# No changes needed - it now uses RandCropByPosNegLabeld with spatial_size [96,96,96]

# Patch-based SWIN-UNETR WITHOUT CoordConv (recommended)
# CoordConv with patches gives inconsistent coordinates - harmful for generalization
swinunetr_patches = deepcopy(swinunetr_cc_patches)
del swinunetr_patches['last_transform'][2]  # Remove CoordConv

# SHORT NAMES for command line convenience:
# p96 = 96³ patches, high negative sampling for DWI artifacts
# Use: -trs p96 -bs 1 -ga 6
# Strategy: Many negative samples to learn DWI artifacts vs real lesions
p96 = deepcopy(unetr_cc)
del p96['first_transform'][2]
del p96['last_transform'][2]  # Remove CoordConv
p96['patches'] = [
    {'RandCropByPosNegLabeld': {
        'keys': ['image', 'label'],
        'label_key': 'label',
        'spatial_size': [96, 96, 96],
        'pos': 1,   # 1 lesion patch
        'neg': 3,   # 3 healthy patches (learn to reject artifacts!)
        'num_samples': 4}},
]

# p64 = 64³ patches, balanced sampling, more spatial diversity
# Use: -trs p64 -bs 2 -ga 3
# Strategy: Smaller patches = more samples per image, learn local patterns
p64 = deepcopy(unetr_cc)
del p64['first_transform'][2]
del p64['last_transform'][2]  # Remove CoordConv
p64['patches'] = [
    {'RandCropByPosNegLabeld': {
        'keys': ['image', 'label'],
        'label_key': 'label',
        'spatial_size': [64, 64, 64],
        'pos': 2,   # 2 lesion patches
        'neg': 4,   # 4 healthy patches (high negative sampling)
        'num_samples': 6}},
]

# p64_sparse = 64³ patches, fewer samples for memory efficiency
# Use: -trs p64_sparse -bs 2 -ga 2
# Strategy: Efficient memory, faster iterations, sees more unique images
p64_sparse = deepcopy(unetr_cc)
del p64_sparse['first_transform'][2]
del p64_sparse['last_transform'][2]  # Remove CoordConv
p64_sparse['patches'] = [
    {'RandCropByPosNegLabeld': {
        'keys': ['image', 'label'],
        'label_key': 'label',
        'spatial_size': [64, 64, 64],
        'pos': 1,   # 1 lesion patch
        'neg': 2,   # 2 healthy patches
        'num_samples': 3}},
]

unetr_no_cc = deepcopy(unetr_cc)
del unetr_no_cc['last_transform'][2]

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
            # 'shear_range': ([(shear_min, shear_max) for _ in range(6)]),
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


# =============================================================================
# MULTI-MODAL TRANSFORM DICT
# =============================================================================
# Literature-grounded augmentations for multi-modal DWI/ADC data.
# See transform_dicts_references.md for scientific citations.
#
# Key features:
# - 'modality_intensity' section: Expanded per-modality by expand_per_modality_transforms
# - Resolution scaling via adapt_transforms_for_resolution
# - Configurable patch size
# =============================================================================

# Base parameters designed for 2mm resolution
_BASE_RESOLUTION = 2

# Default elastic deformation parameters (2mm base)
_ELASTIC_PARAMS_2MM = {
    'sigma_range': (3, 15),
    'magnitude_range': (3, 10),
    'translate_range': (0.5, 3),
}

# Maximum values at 2mm BASE resolution to prevent OOM
# These are scaled by resolution to maintain constant computational cost
# sigma=15 at 2mm → kernel ~91³ (~750K elements), reasonable
_MAX_SIGMA_2MM = 15
_MAX_MAGNITUDE_2MM = 15
_MAX_TRANSLATE_2MM = 5


def _get_resolution_caps(resolution_mm: float) -> tuple:
    """Calculate resolution-aware caps for elastic deformation parameters.

    Maintains approximately constant computational cost regardless of resolution.
    Higher resolution images are larger (N³ scales as 1/res³), so we reduce
    the maximum sigma proportionally to keep kernel computations manageable.

    Formula: max_param(res) = max_param_2mm × (res / 2)

    At 1mm: sigma_max = 7.5, magnitude_max = 7.5, translate_max = 2.5
    At 2mm: sigma_max = 15, magnitude_max = 15, translate_max = 5 (base)
    At 3mm: sigma_max = 22.5, magnitude_max = 22.5, translate_max = 7.5
    """
    scale = resolution_mm / _BASE_RESOLUTION
    return (
        _MAX_SIGMA_2MM * scale,
        _MAX_MAGNITUDE_2MM * scale,
        _MAX_TRANSLATE_2MM * scale,
    )


def create_multimodal_transform_dict(
    resolution_mm: int = 2,
    patch_size: int = 96,
    denoised_data: bool = False,
    num_samples: int = 5,
) -> dict:
    """Create a multi-modal aware transform dictionary.

    Creates a transform dict designed for multi-modal data (e.g., DWI + ADC).
    The 'modality_intensity' section contains transforms that will be expanded
    to per-modality versions by `expand_per_modality_transforms`.

    Parameters
    ----------
    resolution_mm : int
        Image resolution in mm (1, 2, or 3). Elastic deformation parameters
        are scaled accordingly. Default 2mm.
    patch_size : int
        Cubic patch size for RandCropByPosNegLabeld. Default 96.
    denoised_data : bool
        If True, reduces noise augmentation strength (for preprocessed data).
        Default False.

    Returns
    -------
    dict
        Transform dictionary with structure:
        - 'first_transform': Loading, channel-first
        - 'modality_intensity': Per-modality intensity transforms (to be expanded)
        - 'monai_transform': Spatial and shared transforms
        - 'unetr_transform': UNETR-specific augmentations
        - 'last_transform': Final normalization
        - 'patches': Patch sampling

    Notes
    -----
    This dict uses 'modality_intensity' as a placeholder. Call
    `adapt_transforms_for_multimodal` to expand it for actual modalities.

    The function internally calls `adapt_transforms_for_resolution` to scale
    voxel-based parameters.

    See Also
    --------
    adapt_transforms_for_resolution : Scales voxel-based parameters
    expand_per_modality_transforms : Expands modality_intensity section
    transform_dicts_references.md : Literature references

    Examples
    --------
    >>> # For 1mm resolution data with 96^3 patches
    >>> transform_dict = create_multimodal_transform_dict(resolution_mm=1, patch_size=96)

    >>> # For denoised data (reduced augmentation)
    >>> transform_dict = create_multimodal_transform_dict(denoised_data=True)
    """
    # Noise reduction factor for denoised data
    noise_factor = 0.5 if denoised_data else 1.0

    # Scale elastic params for resolution
    # Higher resolution (smaller mm) = more voxels per physical distance
    # So we need MORE voxels: scale = base / target (e.g., 2mm/1mm = 2)
    scale = _BASE_RESOLUTION / resolution_mm

    # Get resolution-aware caps (maintains constant computational cost)
    max_sigma, max_magnitude, max_translate = _get_resolution_caps(resolution_mm)

    # Scale and cap to prevent OOM
    # At 1mm: caps are (7.5, 7.5, 2.5) - same kernel cost as 2mm's (15, 15, 5)
    sigma_range = tuple(
        min(v * scale, max_sigma) for v in _ELASTIC_PARAMS_2MM['sigma_range']
    )
    magnitude_range = tuple(
        min(v * scale, max_magnitude) for v in _ELASTIC_PARAMS_2MM['magnitude_range']
    )
    translate_range = tuple(
        min(v * scale, max_translate) for v in _ELASTIC_PARAMS_2MM['translate_range']
    )

    transform_dict = {
        'first_transform': [
            {'LoadImaged': {'keys': ['image', 'label']}},
            {'EnsureChannelFirstd': {'keys': ['image', 'label']}},
            # Note: MyNormalizeIntensityd moved to modality_intensity
            # to ensure per-modality normalization before concatenation
            {'Binarized': {'keys': ['label'], 'lower_threshold': 0.5}},
        ],

        # Per-modality intensity transforms (expanded by expand_per_modality_transforms)
        # These are applied BEFORE ConcatItemsd to each modality independently
        'modality_intensity': [
            # Normalization MUST be per-modality (no channel_wise support)
            {'MyNormalizeIntensityd': {
                'keys': ['image'],
                'out_min_max': (0, 1),
            }},
            # Histogram shift - different histograms per modality
            {'RandHistogramShiftd': {
                'keys': ['image'],
                'num_control_points': 20,
                'prob': low_prob,
            }},
            # Gibbs ringing - affects both DWI and ADC (propagates through calculation)
            {'RandGibbsNoised': {
                'keys': ['image'],
                'alpha': (0.5, 0.7),
                'prob': high_prob * noise_factor,
            }},
            # K-space spike - affects both (params adjusted per-modality by expand function)
            {'RandKSpaceSpikeNoised': {
                'keys': ['image'],
                'intensity_range': (8, 10),
                'prob': low_prob,  # Adjusted per-modality
            }},
            # Intensity shift - MUST be per-modality (no channel_wise support)
            # DWI and ADC have different intensity distributions
            {'RandShiftIntensityd': {
                'keys': ['image'],
                'offsets': 0.10,
                'prob': high_prob,
            }},
        ],

        # Spatial and shared transforms (applied AFTER ConcatItemsd)
        'monai_transform': [
            # Elastic deformation - same deformation field to all channels
            {'RandAffined': {
                'keys': ['image', 'label'],
                'prob': low_prob,
                'rotate_range': (radians(1), radians(10)),
                'shear_range': ([(shear_min, shear_max) for _ in range(6)]),
                'translate_range': translate_range,
                'scale_range': (0.02, 0.15),
                'mode': ('bilinear', 'nearest'),
                'padding_mode': 'reflection',
            }},
            # Rician noise - HAS channel_wise support, can stay here
            {'RandRicianNoised': {
                'keys': ['image'],
                'prob': low_prob * noise_factor,
                'mean': 0.1,
                'std': 0.025 * noise_factor,
                'channel_wise': True,  # Independent noise per channel
            }},
            # Bias field - different pattern per channel (acceptable)
            {'RandBiasFieldd': {
                'keys': ['image'],
                'prob': high_prob * noise_factor,
                'coeff_range': (0.0, 0.05 * noise_factor),
            }},
        ],

        'unetr_transform': [
            {'RandFlipd': {
                'keys': ['image', 'label'],
                'spatial_axis': [0],
                'prob': low_prob,
            }},
            # NOTE: RandShiftIntensityd moved to modality_intensity
            # It has no channel_wise support, so must be applied per-modality
        ],

        'last_transform': [
            # Reactivate if needed:
            # {'Binarized': {
            #     'keys': ['label'],
            #     'lower_threshold': 0.25,
            # }},
            # Final normalization after all augmentations
            {'MyNormalizeIntensityd': {
                'keys': ['image'],
                'out_min_max': (0, 1),
            }},
        ],

        'patches': [
            {'RandCropByPosNegLabeld': {
                'keys': ['image', 'label'],
                'label_key': 'label',
                'spatial_size': [patch_size, patch_size, patch_size],
                'pos': 1,
                'neg': 1,  # High negative sampling for artifact rejection
                'num_samples': num_samples,
            }},
        ],
    }

    return transform_dict


# Convenience aliases for common configurations
def mm1_p96(**kwargs):
    """1mm resolution, 96^3 patches."""
    return create_multimodal_transform_dict(resolution_mm=1, patch_size=96, **kwargs)


def mm1_p64(**kwargs):
    """1mm resolution, 64^3 patches."""
    return create_multimodal_transform_dict(resolution_mm=1, patch_size=64, **kwargs)


def mm2_p96(**kwargs):
    """2mm resolution (default), 96^3 patches."""
    return create_multimodal_transform_dict(resolution_mm=2, patch_size=96, **kwargs)


def _with_patch_augment(td: dict) -> dict:
    """Move RandAffined from monai_transform to a patch_augment section after patches.

    Affine augmentation on a 96³ patch is ~25x faster than on the full 182³ volume.
    Each patch is augmented independently via MONAI Compose map_items.
    """
    affine = [e for e in td['monai_transform'] if 'RandAffined' in e]
    td['monai_transform'] = [e for e in td['monai_transform'] if 'RandAffined' not in e]
    td['patch_augment'] = affine
    return td


def mm1_p96_pa(**kwargs):
    """1mm resolution, 96^3 patches, affine augmentation applied at patch level."""
    return _with_patch_augment(create_multimodal_transform_dict(resolution_mm=1, patch_size=96, **kwargs))


def mm1_p64_pa(**kwargs):
    """1mm resolution, 64^3 patches, affine augmentation applied at patch level."""
    return _with_patch_augment(create_multimodal_transform_dict(resolution_mm=1, patch_size=64, **kwargs))


def mm2_p96_pa(**kwargs):
    """2mm resolution, 96^3 patches, affine augmentation applied at patch level."""
    return _with_patch_augment(create_multimodal_transform_dict(resolution_mm=2, patch_size=96, **kwargs))


def _with_patch_gibbs_no_spike(td: dict) -> dict:
    """Move affine + Gibbs to patch level; remove KSpaceSpike entirely.

    Extends _with_patch_augment by also:
    - Removing RandKSpaceSpikeNoised from modality_intensity (full-image herringbone
      pattern is qualitatively wrong at patch scale and not used by nnUNet).
    - Moving RandGibbsNoised from modality_intensity to patch_augment (edge-ringing
      approximation is acceptable at patch scale; saves ~232 MB FFT workspace per worker).

    In patch_augment, Gibbs uses keys=['image'] on the concatenated (N_ch, patch³) tensor.
    MONAI applies the same alpha to all channels, which is physically sensible (same
    k-space truncation for co-registered modalities).
    """
    td = _with_patch_augment(td)
    gibbs = [e for e in td['modality_intensity'] if 'RandGibbsNoised' in e]
    td['modality_intensity'] = [
        e for e in td['modality_intensity']
        if 'RandGibbsNoised' not in e and 'RandKSpaceSpikeNoised' not in e
    ]
    td['patch_augment'] = td.get('patch_augment', []) + gibbs
    return td


def mm1_p96_pag(**kwargs):
    """1mm resolution, 96^3 patches, affine + Gibbs at patch level, no KSpace spike."""
    return _with_patch_gibbs_no_spike(
        create_multimodal_transform_dict(resolution_mm=1, patch_size=96, **kwargs)
    )


def mm1_p64_pag(**kwargs):
    """1mm resolution, 64^3 patches, affine + Gibbs at patch level, no KSpace spike."""
    return _with_patch_gibbs_no_spike(
        create_multimodal_transform_dict(resolution_mm=1, patch_size=64, **kwargs)
    )


def mm2_p96_pag(**kwargs):
    """2mm resolution, 96^3 patches, affine + Gibbs at patch level, no KSpace spike."""
    return _with_patch_gibbs_no_spike(
        create_multimodal_transform_dict(resolution_mm=2, patch_size=96, **kwargs)
    )
