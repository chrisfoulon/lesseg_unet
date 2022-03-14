from math import radians
from copy import deepcopy
high_prob = .2
low_prob = .1
tiny_prob = 0.05
# high_prob = 1
# low_prob = 1
# tiny_prob = 1
def_spatial_size = [96, 128, 96]
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

unetr_cc = {
    'first_transform': [
        {'LoadImaged': {
            'keys': ['image', 'label']}},
        {'AddChanneld': {'keys': ['image', 'label']}},
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
            'num_control_points': (10, 15),
            'prob': low_prob}
         },
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
        {'PrintDim': {'keys': ['image', 'label'], 'msg': 'PrintDim before ToTensord'}},
        {'ToTensord': {'keys': ['image', 'label']}},
        {'RandomBiasField': {
            'include': ['image'],
            'p': low_prob,
            'coefficients': 0.1}
         },
    ],
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
        {'ToTensord': {'keys': ['image', 'label']}},
        {'CoordConvd': {'keys': ['image']}}
    ],
}

