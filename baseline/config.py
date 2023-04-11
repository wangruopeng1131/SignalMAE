ResNet50 = {
    'head_config': {
        'in_channels': 1,
        'out_channels': 64,
        'kernel_size': 7,
        'stride': 2,
        'padding': 3,
        'max_pool_kernel': 3,
        'max_pool_stride': 2,
        'max_pool_padding': 1
    },
    'hidden_layers': {
        '1_block': {
            # BottleNeck(64, 64, 256, kernel_size, downSampling=False),
            # BottleNeck(256, 64, 256, kernel_size, downSampling=False),
            # BottleNeck(256, 64, 256, kernel_size, downSampling=False),
            '1_layers': {
                'in_channels': 64,
                'hidden_channels': 64,
                'out_channels': 256,
                'kernel_size': [1, 3, 1],
                'downSampling': False
            },
            '2_layers': {
                'in_channels': 256,
                'hidden_channels': 64,
                'out_channels': 256,
                'kernel_size': [1, 3, 1],
                'downSampling': False
            },
            '3_layers': {
                'in_channels': 256,
                'hidden_channels': 64,
                'out_channels': 256,
                'kernel_size': [1, 3, 1],
                'downSampling': False
            }
        },
        '2_block': {
            # BottleNeck(256, 128, 512, kernel_size, downSampling=True),
            # BottleNeck(512, 128, 512, kernel_size, downSampling=False),
            # BottleNeck(512, 128, 512, kernel_size, downSampling=False),
            # BottleNeck(512, 128, 512, kernel_size, downSampling=False),
            '1_layers': {
                'in_channels': 256,
                'hidden_channels': 128,
                'out_channels': 512,
                'kernel_size': [1, 3, 1],
                'downSampling': True
            },
            '2_layers': {
                'in_channels': 512,
                'hidden_channels': 128,
                'out_channels': 512,
                'kernel_size': [1, 3, 1],
                'downSampling': False
            },
            '3_layers': {
                'in_channels': 512,
                'hidden_channels': 128,
                'out_channels': 512,
                'kernel_size': [1, 3, 1],
                'downSampling': False
            },
            '4_layers': {
                'in_channels': 512,
                'hidden_channels': 128,
                'out_channels': 512,
                'kernel_size': [1, 3, 1],
                'downSampling': False
            }
        },
        '3_block': {
            # BottleNeck(512, 256, 1024, kernel_size, downSampling=True),
            # BottleNeck(1024, 256, 1024, kernel_size, downSampling=False),
            # BottleNeck(1024, 256, 1024, kernel_size, downSampling=False),
            # BottleNeck(1024, 256, 1024, kernel_size, downSampling=False),
            # BottleNeck(1024, 256, 1024, kernel_size, downSampling=False),
            # BottleNeck(1024, 256, 1024, kernel_size, downSampling=False),
            '1_layers': {
                'in_channels': 512,
                'hidden_channels': 256,
                'out_channels': 1024,
                'kernel_size': [1, 3, 1],
                'downSampling': True
            },
            '2_layers': {
                'in_channels': 1024,
                'hidden_channels': 256,
                'out_channels': 1024,
                'kernel_size': [1, 3, 1],
                'downSampling': False
            },
            '3_layers': {
                'in_channels': 1024,
                'hidden_channels': 256,
                'out_channels': 1024,
                'kernel_size': [1, 3, 1],
                'downSampling': False
            },
            '4_layers': {
                'in_channels': 1024,
                'hidden_channels': 256,
                'out_channels': 1024,
                'kernel_size': [1, 3, 1],
                'downSampling': False
            },
            '5_layers': {
                'in_channels': 1024,
                'hidden_channels': 256,
                'out_channels': 1024,
                'kernel_size': [1, 3, 1],
                'downSampling': False
            }
        },
        '4_block': {
            # BottleNeck(1024, 512, 2048, kernel_size, downSampling=True),
            # BottleNeck(2048, 512, 2048, kernel_size, downSampling=False),
            # BottleNeck(2048, 512, 2048, kernel_size, downSampling=False),
            '1_layers': {
                'in_channels': 1024,
                'hidden_channels': 512,
                'out_channels': 2048,
                'kernel_size': [1, 3, 1],
                'downSampling': True
            },
            '2_layers': {
                'in_channels': 2048,
                'hidden_channels': 512,
                'out_channels': 2048,
                'kernel_size': [1, 3, 1],
                'downSampling': False
            },
            '3_layers': {
                'in_channels': 2048,
                'hidden_channels': 512,
                'out_channels': 2048,
                'kernel_size': [1, 3, 1],
                'downSampling': False
            }
        }
    },
    'out_channels': 2048,
    'classes': 2
}


ResNet18 = {
    'head_config': {
        'in_channels': 1,
        'out_channels': 64,
        'kernel_size': 7,
        'stride': 2,
        'padding': 3,
        'max_pool_kernel': 3,
        'max_pool_stride': 2,
        'max_pool_padding': 1
    },
    'hidden_layers': {
        '1_block': {
            # BottleNeck(64, 64, 256, kernel_size, downSampling=False),
            # BottleNeck(256, 64, 256, kernel_size, downSampling=False),
            '1_layers': {
                'in_channels': 64,
                'hidden_channels': None,
                'out_channels': 64,
                'kernel_size': [3, 3],
                'downSampling': False
            },
            '2_layers': {
                'in_channels': 64,
                'hidden_channels': None,
                'out_channels': 64,
                'kernel_size': [3, 3],
                'downSampling': False
            }
        },
        '2_block': {
            '1_layers': {
                'in_channels': 64,
                'hidden_channels': None,
                'out_channels': 128,
                'kernel_size': [3, 3],
                'downSampling': True
            },
            '2_layers': {
                'in_channels': 128,
                'hidden_channels': None,
                'out_channels': 128,
                'kernel_size': [3, 3],
                'downSampling': False
            }
        },
        '3_block': {
            '1_layers': {
                'in_channels': 128,
                'hidden_channels': None,
                'out_channels': 256,
                'kernel_size': [3, 3],
                'downSampling': True
            },
            '2_layers': {
                'in_channels': 256,
                'hidden_channels': None,
                'out_channels': 256,
                'kernel_size': [3, 3],
                'downSampling': False
            }
        },
        '4_block': {
            '1_layers': {
                'in_channels': 256,
                'hidden_channels': None,
                'out_channels': 512,
                'kernel_size': [3, 3],
                'downSampling': True
            },
            '2_layers': {
                'in_channels': 512,
                'hidden_channels': None,
                'out_channels': 512,
                'kernel_size': [3, 3],
                'downSampling': False
            },
        }
    },
    'out_channels': 512,
    'classes': 4
}