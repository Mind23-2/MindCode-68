import os
import ast
import random
import argparse
import numpy as np
from pathlib import Path

import mindspore
import mindspore.nn as nn
from mindspore import context
from mindspore import Parameter
from mindspore import Tensor
from mindspore import dataset as de
from mindspore.common import set_seed
from mindspore.nn.optim.momentum import Momentum
import mindspore.common.initializer as weight_init
from mindspore.communication.management import init
from mindspore.train.model import Model, ParallelMode
from mindspore.train.callback import SummaryCollector
from mindspore.train.callback import (ModelCheckpoint, CheckpointConfig,
                                      LossMonitor, TimeMonitor)
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.lr import get_lr
from src.ResNet3D import generate_model
from src.save_callback import SaveCallback
from src.loss import SoftmaxCrossEntropyExpand, CrossEntropySmooth
from src.dataset import create_train_dataset, create_eval_dataset


parser = argparse.ArgumentParser()

parser.add_argument('--device_target', default='Ascend', type=str, help='Device target')

parser.add_argument('--device_id', default=0, type=int)

parser.add_argument('--is_modelarts', type=ast.literal_eval, default=False)

parser.add_argument('--run_distribute', type=ast.literal_eval, default=False, help='Run distribute')

parser.add_argument('--data_url', type=str, default=None, help="Used when is_modelarts is True")

parser.add_argument('--train_url', type=str, default=None,
                    help="Used when is_modelarts is True. Train output in modelarts")

parser.add_argument('--dataset', type=str)

args_opt = parser.parse_args()

if args_opt.is_modelarts:
    import moxing as mox

if args_opt.dataset == 'ucf101':
    from src.config import config_ucf101 as cfg
elif args_opt.dataset == 'hmdb51':
    from src.config import config_hmdb51 as cfg
else:
    raise NotImplementedError

random.seed(1)
np.random.seed(1)
de.config.set_seed(1)
set_seed(1)

if __name__ == '__main__':
    target = args_opt.device_target

    # init context
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)
    if args_opt.run_distribute:
        if target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(device_id=device_id, enable_auto_mixed_precision=True)
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
            init()
    else:
        if target == "Ascend":
            device_id = args_opt.device_id
            context.set_context(device_id=device_id)

    # profiler = Profiler()

    # create dataset
    if args_opt.is_modelarts:
        root_path = '/cache/mmq_' + os.getenv('DEVICE_ID')
        mox.file.copy_parallel(src_url=args_opt.data_url, dst_url=root_path)
        unzip_command = "unzip -o -q " + root_path + '/jpg.zip' \
                                                     " -d " + root_path + '/jpg'
        os.system(unzip_command)
        cfg.video_path = Path(root_path + "/jpg")
        cfg.annotation_path = Path(root_path + '/json/ucf101_01.json')
        cfg.pretrain_path = root_path + "/*.ckpt"
        cfg.result_path = root_path + '/train_output/'

    train_dataset = create_train_dataset(cfg.video_path, cfg.annotation_path, cfg,
                                         batch_size=cfg.batch_size, target="Ascend")
    if cfg.eval_in_training:
        inference_dataset = create_eval_dataset(cfg.video_path, cfg.annotation_path, cfg)

    step_size = train_dataset.get_dataset_size()

    # load pre_trained model and define net
    net = generate_model(n_classes=cfg.n_classes, stop_weights_update=True)
    param_dict = load_checkpoint(cfg.pretrain_path)
    load_param_into_net(net, param_dict)

    for _, cell in net.cells_and_names():
        if isinstance(cell, nn.Dense):
            weight_shape = cell.weight.shape
            stdv = np.sqrt(1. / weight_shape[1])
            fc_weight = np.random.uniform(-stdv, stdv, weight_shape).astype(np.float32)
            fc_bias = np.random.uniform(-stdv, stdv, (weight_shape[0],)).astype(np.float32)

            cell.weight.set_data(Tensor(fc_weight, mindspore.float32))
            cell.bias.set_data(Tensor(fc_bias, mindspore.float32))

    # init lr
    lr = get_lr(lr_init=cfg.lr_init, lr_end=cfg.lr_end, lr_max=cfg.lr_max,
                warmup_epochs=cfg.warmup_epochs, total_epochs=cfg.n_epochs,
                steps_per_epoch=step_size, lr_decay_mode=cfg.lr_decay_mode)
    print("[lr] : ", lr)

    # define opt
    param_to_train = []
    if cfg.start_ft == 'conv1':
        param_to_train = net.trainable_params()
    else:
        start = False
        for param in net.trainable_params():
            if not start and cfg.start_ft in param.name:
                start = True
            if start:
                param_to_train.append(param)
    
    print(param_to_train)

    net_opt = nn.SGD(param_to_train, lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay, nesterov=False)
    # net_opt = Momentum(param_to_train, lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    # define loss, model
    # loss = SoftmaxCrossEntropyExpand(sparse=True)
    # loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    loss = CrossEntropySmooth(smooth_factor=0.1, num_classes=cfg.n_classes)
    model = Model(net, loss_fn=loss, optimizer=net_opt)

    # define callbacks
    cb = []
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_epochs * step_size,
                                 keep_checkpoint_max=cfg.keep_checkpoint_max)
    ckpt_cb = ModelCheckpoint(prefix="resnet-3d", directory=cfg.result_path, config=config_ck)

    if cfg.eval_in_training:
        save_cb = SaveCallback(model, inference_dataset, 20, cfg)
        cb.append(save_cb)

    # summary_dir = cfg.result_path + "summary_dir_" + str(os.getenv('DEVICE_ID'))
    # if not os.path.exists(summary_dir):
    #     os.mkdir(summary_dir)
    # summary_collector = SummaryCollector(summary_dir=summary_dir, collect_freq=1)

    cb += [time_cb, loss_cb, ckpt_cb]
    print("==============================config======================\n")
    for item in cfg:
        print(item, ' : ', cfg[item])
    print("=======Training Begin========")
    model.train(cfg.n_epochs, train_dataset, callbacks=cb, dataset_sink_mode=False)

    if args_opt.is_modelarts:
        mox.file.copy_parallel(src_url=cfg.result_path, dst_url=args_opt.train_url)
    # profiler.analyse()
