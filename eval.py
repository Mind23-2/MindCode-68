import os
import ast
import time
import random
import argparse
import numpy as np

import mindspore.nn as nn
from mindspore import context
from mindspore import dataset as de
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from mindspore.train.model import Model

from src.dataset import create_eval_dataset
from src.ResNet3D import generate_model
from src.inference import (Inference, load_ground_truth, load_result,
                           remove_nonexistent_ground_truth, calculate_clip_acc)


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


class NetWithSoftmax(nn.Cell):
    def __init__(self, net):
        super(NetWithSoftmax, self).__init__()
        self.softmax = nn.Softmax()
        self.net = net

    def construct(self, x):
        out = self.net(x)
        out = self.softmax(out)
        return out


if __name__ == '__main__':
    t1 = time.time()
    target = args_opt.device_target
    # init context
    device_id = args_opt.device_id
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False,
                        device_id=device_id)

    net = generate_model(n_classes=cfg.n_classes, no_max_pool=False)
    param_dict = load_checkpoint(cfg.inference_ckpt_path)  # need to be modified
    load_param_into_net(net, param_dict)
    
    net = NetWithSoftmax(net)
    net.set_train(False)

    model = Model(net)

    predict_data = create_eval_dataset(cfg.video_path, cfg.annotation_path, cfg)
    inference = Inference()
    inference_results, clip_inference_results = inference(predict_data, model, cfg.annotation_path)

    print('load ground truth')
    ground_truth, class_labels_map = load_ground_truth(cfg.annotation_path, "validation")
    print('number of ground truth: {}'.format(len(ground_truth)))

    n_ground_truth_top_1 = len(ground_truth)
    n_ground_truth_top_5 = len(ground_truth)

    result_top1, result_top5 = load_result(clip_inference_results, class_labels_map)

    # print("==================result_top1===========\n", result_top1)

    ground_truth_top1 = remove_nonexistent_ground_truth(ground_truth, result_top1)
    ground_truth_top5 = remove_nonexistent_ground_truth(ground_truth, result_top5)

    if cfg.ignore:
        n_ground_truth_top_1 = len(ground_truth_top1)
        n_ground_truth_top_5 = len(ground_truth_top5)

    correct_top1 = [1 if line[1] in result_top1[line[0]] else 0 for line in ground_truth_top1]
    correct_top5 = [1 if line[1] in result_top5[line[0]] else 0 for line in ground_truth_top5]

    clip_acc = calculate_clip_acc(inference_results, ground_truth, class_labels_map)
    accuracy_top1 = float(sum(correct_top1)) / float(n_ground_truth_top_1)
    accuracy_top5 = float(sum(correct_top5)) / float(n_ground_truth_top_5)
    print('==================Accuracy=================\n'
          ' clip-acc : {} \ttop-1 : {} \ttop-5: {}'.format(clip_acc, accuracy_top1, accuracy_top5))
    t2 = time.time()
    print("Total time : {} s".format(t2 - t1))
