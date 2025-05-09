import torch.nn as nn

pooling_layer_map = {
    "MaxPool2d": {
        "module": nn.MaxPool2d,
        "params": {
            "kernel_size": 2,
            "stride": 2,
            "padding": 0,
        }
    },
    "AvgPool2d": {
        "module": nn.AvgPool2d,
        "params": {
            "kernel_size": 2,
            "stride": 2,
            "padding": 0,
        }
    },
    "AdaptiveMaxPool2d": {
        "module": nn.AdaptiveMaxPool2d,
        "params": {
            "output_size": (1, 1),
        }
    },
    "AdaptiveAvgPool2d": {
        "module": nn.AdaptiveAvgPool2d,
        "params": {
            "output_size": (1, 1),
        }
    }
}

activation_layer_map = {
    "ReLU": {"module": nn.ReLU, "params": {}},
    "LeakyReLU": {"module": nn.LeakyReLU, "params": {"negative_slope": 0.01}},
}

batch_norm_layer_map = {
    "BatchNorm2d": {
        "module": nn.BatchNorm2d,
        "params": {},
    },
}

dropout_layer_map = {
    "Dropout": {"module": nn.Dropout, "params": {}},
}
