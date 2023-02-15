from easydict import EasyDict
from pathlib import Path

config_ucf101 = EasyDict({
    'video_path': Path('/opt_data/xidian_wks/mmq/resnet-3d/dataset/ucf101/jpg/'),
    'annotation_path': Path('/opt_data/xidian_wks/mmq/resnet-3d/dataset/ucf101/json/ucf101_01.json'),
    'result_path': './results/ucf101',
    'pretrain_path': '/opt_data/xidian_wks/mmq/resnet-3d/pre_trained_ckpt/r3d50_KM_200ep.ckpt',
    'inference_ckpt_path': '/opt_data/xidian_wks/mmq/resnet/',
    'n_classes': 101,
    'sample_size': 112,
    'sample_duration': 16,
    'sample_t_stride': 1,
    'train_crop': 'center',
    'colorjitter': False,
    'train_crop_min_scale': 0.25,
    'train_crop_min_ratio': 0.75,
    'train_t_crop': 'random',
    'inference_stride': 16,
    'ignore': True,
    'start_ft': 'layer4',  # choices = [conv1, layer1, layer2, layer3, layer4, fc]
    'loss_scale': 1024,
    'momentum': 0.9,
    'weight_decay': 0.001,
    'batch_size': 128,
    'n_epochs': 200,
    'save_checkpoint_epochs': 5,
    'keep_checkpoint_max': 10,
    'lr_decay_mode': 'poly',
    'warmup_epochs': 5,
    'lr_init': 0,
    'lr_max': 0.003,
    'lr_end': 0,
    'eval_in_training': True
})

config_hmdb51 = EasyDict({
    'video_path': Path('/opt_data/xidian_wks/mmq/resnet-3d/dataset/hmdb51/jpg/'),
    'annotation_path': Path('/opt_data/xidian_wks/mmq/resnet-3d/dataset/hmdb51/json/hmdb51_1.json'),
    'result_path': './results/hmdb51',
    'pretrain_path': '/opt_data/xidian_wks/mmq/resnet-3d/pre_trained_ckpt/r3d50_KM_200ep.ckpt',
    'inference_ckpt_path': "/opt_data/xidian_wks/mmq/resnet-3d/scripts/device0/results/hmdb51/resnet-3d-200_6.ckpt",
    'n_classes': 51,
    'sample_size': 112,
    'sample_duration': 16,
    'sample_t_stride': 1,
    'train_crop': 'center',
    'colorjitter': False,
    'train_crop_min_scale': 0.25,
    'train_crop_min_ratio': 0.75,
    'train_t_crop': 'random',
    'inference_stride': 16,
    'ignore': True,
    'start_ft': 'layer4',  # choices = [conv1, layer1, layer2, layer3, layer4, fc]
    'loss_scale': 1024,
    'momentum': 0.9,
    'weight_decay': 0.001,
    'batch_size': 128,
    'n_epochs': 200,
    'save_checkpoint_epochs': 5,
    'keep_checkpoint_max': 10,
    'lr_decay_mode': 'poly',  # choices = [steps, poly, cosine, linear]
    'warmup_epochs': 10,
    'lr_init': 0,
    'lr_max': 0.001,
    'lr_end': 0,
    'eval_in_training': True
})
