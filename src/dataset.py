import os
from pathlib import Path
import math

import mindspore.dataset as ds
import mindspore.common.dtype as mstype
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
from mindspore.communication.management import init, get_rank, get_group_size

from .videodataset import DatasetGenerator
from .videodataset_multiclips import DatasetGeneratorMultiClips
from .temporal_transforms import (LoopPadding, TemporalRandomCrop,
                                  TemporalCenterCrop, TemporalEvenCrop,
                                  SlidingWindow, TemporalSubsampling)
from .temporal_transforms import Compose as TemporalCompose
from .pil_transforms import PILTrans, EvalPILTrans


class MySampler:
    def __init__(self, dataset, local_rank, world_size):
        self.__num_data = len(dataset)
        self.__local_rank = local_rank
        self.__world_size = world_size
        self.samples_per_rank = int(math.ceil(self.__num_data / float(self.__world_size)))
        self.total_num_samples = self.samples_per_rank * self.__world_size

    def __iter__(self):
        indices = list(range(self.__num_data))
        indices.extend(indices[:self.total_num_samples - len(indices)])
        indices = indices[self.__local_rank:self.total_num_samples:self.__world_size]
        return iter(indices)

    def __len__(self):
        return self.samples_per_rank


def create_train_dataset(root_path, annotation_path, opt, repeat_num=1, batch_size=32, target="Ascend"):
    if target == "Ascend":
        device_num, rank_id = _get_rank_info()

    assert opt.train_t_crop in ['random', 'center']
    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    if opt.train_t_crop == 'random':
        temporal_transform.append(TemporalRandomCrop(opt.sample_duration))
    elif opt.train_t_crop == 'center':
        temporal_transform.append(TemporalCenterCrop(opt.sample_duration))
    temporal_transform = TemporalCompose(temporal_transform)

    train_dataset = DatasetGenerator(root_path=root_path, annotation_path=annotation_path,
                                     subset='training', temporal_transform=temporal_transform)
    sampler = MySampler(dataset=train_dataset, local_rank=rank_id, world_size=device_num)

    if device_num == 1:
        dataset = ds.GeneratorDataset(train_dataset, column_names=["data", "label"],
                                      num_parallel_workers=4, shuffle=True, )
    else:
        dataset = ds.GeneratorDataset(train_dataset, column_names=["data", "label"], sampler=sampler,
                                      num_parallel_workers=4, shuffle=True, num_shards=device_num,
                                      shard_id=rank_id)
    type_cast_op = C2.TypeCast(mstype.int32)
    mean = [0.4345, 0.4051, 0.3775]
    std = [0.2768, 0.2713, 0.2737]

    # mean = [0.4477, 0.4209, 0.3906]
    # std = [0.2767, 0.2695, 0.2714]

    # mean = [0.5, 0.5, 0.5]
    # std = [0.5, 0.5, 0.5]

    trans = PILTrans(opt, mean=mean, std=std)
    dataset = dataset.map(operations=type_cast_op, input_columns='label', num_parallel_workers=4)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True, per_batch_map=trans, num_parallel_workers=4,
                            input_columns=['data', 'label'])
    dataset = dataset.repeat(repeat_num)
    return dataset


def create_eval_dataset(root_path, annotation_path, opt, target="Ascend"):
    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    temporal_transform.append(
        SlidingWindow(opt.sample_duration, opt.inference_stride))
    temporal_transform = TemporalCompose(temporal_transform)
    eval_dataset = DatasetGeneratorMultiClips(root_path=root_path, annotation_path=annotation_path,
                                              subset='validation', temporal_transform=temporal_transform,
                                              target_type=['video_id', 'segment'])

    dataset = ds.GeneratorDataset(eval_dataset, column_names=["data", "label"],
                                  num_parallel_workers=4, shuffle=False, )

    type_cast_op = C2.TypeCast(mstype.int32)
    mean = [0.4345, 0.4051, 0.3775]
    std = [0.2768, 0.2713, 0.2737]

    # mean = [0.4477, 0.4209, 0.3906]
    # std = [0.2767, 0.2695, 0.2714]

    # mean = [0.5, 0.5, 0.5]
    # std = [0.5, 0.5, 0.5]

    trans = EvalPILTrans(opt, mean=mean, std=std)
    dataset = dataset.map(operations=type_cast_op, input_columns='label')
    dataset = dataset.batch(batch_size=1, drop_remainder=True, per_batch_map=trans, input_columns=['data', 'label'],
                            num_parallel_workers=4)
    return dataset


def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = 1
        rank_id = 0

    return rank_size, rank_id


def test():
    root_path = Path('/opt_data/xidian_wks/mmq/resnet-3d/dataset/ucf101/jpg/')
    annotation_path = Path("/opt_data/xidian_wks/mmq/resnet-3d/dataset/ucf101/json/ucf101_01.json")
    from .opts import parse_opts
    import time
    dataset = create_train_dataset(root_path, annotation_path, parse_opts())
    step_size = dataset.get_dataset_size()
    for step, data in enumerate(dataset.create_dict_iterator()):
        print("------------{} / {}----------".format(step, step_size))
        print(
            '=====data shape:\n{}\n'.format(data["data"].shape),
            '=====label\n{}\n'.format(data["label"]),
            '=====data :\n{}\n'.format(data["data"])
        )
