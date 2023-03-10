import numpy as np

import mindspore.dataset.vision.py_transforms as py_trans
from mindspore.dataset.transforms.py_transforms import Compose


class PILTrans:
    def __init__(self, opt, mean, std):
        super(PILTrans).__init__()
        self.to_pil = py_trans.ToPIL()
        self.random_resized_crop = \
            py_trans.RandomResizedCrop(opt.sample_size, scale=(opt.train_crop_min_scale, 1.0),
                                       ratio=(opt.train_crop_min_ratio, 1.0 / opt.train_crop_min_ratio))
        self.random_horizontal_flip = py_trans.RandomHorizontalFlip(prob=0.5)
        self.color = py_trans.RandomColorAdjust(0.4, 0.4, 0.4, 0.1)
        self.normalize = py_trans.Normalize(mean=mean, std=std)
        self.to_tensor = py_trans.ToTensor()
        self.resize = py_trans.Resize(opt.sample_size)
        self.center_crop = py_trans.CenterCrop(opt.sample_size)
        self.opt = opt

    def __call__(self, data, labels, batchInfo):
        data_ret = []
        for _, imgs in enumerate(data):
            imgs_ret = []
            for idx in range(0, 16):
                img = imgs[idx]
                img_pil = self.to_pil(img)
                if self.opt.train_crop == 'random':
                    img_pil = self.random_resized_crop(img_pil)
                else:
                    img_pil = self.resize(img_pil)
                    img_pil = self.center_crop(img_pil)
                img_pil = self.random_horizontal_flip(img_pil)
                if self.opt.colorjitter:
                    img_pil = self.color(img_pil)
                img_array = self.to_tensor(img_pil)
                img_array = self.normalize(img_array)
                imgs_ret.append(img_array)
            imgs_ret = np.array(imgs_ret)
            imgs_ret = imgs_ret.transpose((1, 0, 2, 3))  # DCHW -> CDHW
            data_ret.append(imgs_ret)

        return data_ret, labels


class EvalPILTrans:
    def __init__(self, opt, mean, std):
        super(EvalPILTrans).__init__()
        self.to_pil = py_trans.ToPIL()
        self.resize = py_trans.Resize(opt.sample_size)
        self.center_crop = py_trans.CenterCrop(opt.sample_size)
        self.normalize = py_trans.Normalize(mean=mean, std=std)
        self.to_tensor = py_trans.ToTensor()

    def __call__(self, data, labels, batchInfo):
        data = data[0]
        N = data.shape[0]
        D = data.shape[1]
        data_tmp = []
        data_ret = []
        for i in range(0, N):
            video_ret = []
            video = data[i]
            for j in range(0, D):
                img = video[j]
                img_pil = self.to_pil(img)
                img_pil = self.resize(img_pil)
                img_pil = self.center_crop(img_pil)
                img_array = self.to_tensor(img_pil)
                img_array = self.normalize(img_array)
                video_ret.append(img_array)
            video_ret = np.array(video_ret)
            video_ret = video_ret.transpose(1, 0, 2, 3)
            data_tmp.append(video_ret)
        data_ret.append(data_tmp)
        return data_ret, labels
