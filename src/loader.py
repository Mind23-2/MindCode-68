from PIL import Image
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.vision.py_transforms as py_trans
import mindspore.dataset.vision.c_transforms as c_trans


class ImageLoaderPIL(object):

    def __call__(self, path):
        with path.open('rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')


class VideoLoader(object):

    def __init__(self, image_name_formatter, image_loader=None):
        self.image_name_formatter = image_name_formatter
        if image_loader is None:
            self.image_loader = ImageLoaderPIL()
        else:
            self.image_loader = image_loader

    def __call__(self, video_path, frame_indices):
        video = []
        for i in frame_indices:
            image_path = video_path / self.image_name_formatter(i)
            if image_path.exists():
                img = np.array(self.image_loader(image_path))
                video.append(img)
        return np.array(video)