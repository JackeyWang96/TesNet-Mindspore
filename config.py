"""
network config setting, will be used in train.py and eval.py
"""
from easydict import EasyDict as ed

# config for resnet50, imagenet2012
config_cub = ed({
    "class_num": 200,
    "img_size" : 224,
    "batch_size": 32,
    "momentum": 0.875,
    "weight_decay": 1/32768,
    "label_smooth_factor": 0.1,
    "epoch_size": 90,
    "lr": 0.256,
    "lr_end": 0.0,
    "warmup": 1,
    "save_checkpoint": True,
    "save_checkpoint_epochs": 5,
    "keep_checkpoint_max": 20,
    "save_checkpoint_path": "./saved_model/",

    "fixed_lr": "False",
    #tesnet coef
    "base_architecture": "vgg19",
    #"prototype_shape" : (2000, 64, 1, 1),
    "prototype_activation_function" : 'log',
    "add_on_layers_type" : 'regular',

    #dataset
    "data_path" : '/data/wangjiaqi/datasets/CUB_200_2011/CUB_200_2011/new_datasets/cub200_cropped/',
    "train_dir" : '/data/wangjiaqi/datasets/CUB_200_2011/CUB_200_2011/new_datasets/cub200_cropped/train_cropped_augmented/',
    "test_dir" : '/data/wangjiaqi/datasets/CUB_200_2011/CUB_200_2011/new_datasets/cub200_cropped/test_cropped/',
    "train_push_dir" : '/data/wangjiaqi/datasets/CUB_200_2011/CUB_200_2011/new_datasets/cub200_cropped/train_cropped/',

    "train_batch_size" : 80,
    "test_batch_size" : 100,
    "train_push_batch_size" :75,
    #train
    "joint_optimizer_lrs" : {'features': 1e-4, 'add_on_layers': 3e-3,'prototype_vectors': 3e-3},
    "joint_lr_step_size" : 5,

    "warm_optimizer_lrs" : {'add_on_layers': 3e-3,'prototype_vectors': 3e-3},

    "last_layer_optimizer_lr" : 1e-4,
    "coefs" : {
        'crs_ent': 1,
        'clst': 0.8,
        'sep': -0.08,
        'l1': 1e-4,
        'orth': 1e-4,
        'sub_sep': -1e-7,
    },
    "num_train_epochs" : 10,
    "num_warm_epochs" : 5,

    "push_start" : 10,
    "push_epochs" : [i for i in range(20) if i % 10 == 0], #根据train_epochs来计算的

})