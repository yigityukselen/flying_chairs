import torch
import torchvision
import utils.frame_utils
import os, math, random
import numpy as np
from os.path import *
from glob import glob

class Args():
    def __init__(self, crop_size, inference_size):
        self.crop_size = crop_size
        self.inference_size = inference_size

class StaticRandomCrop():
    def __init__(self, image_size, crop_size):
        self.crop_height, self.crop_width = crop_size
        height, width = image_size
        self.random_height = random.randint(0, height - self.crop_height)
        self.random_width = random.randint(0, width - self.crop_width)

    def __call__(self, image):
        return image[self.random_height:(self.random_height + self.crop_height), self.random_width:(self.random_width + self.crop_width), :]

class StaticCenterCrop():
    def __init__(self, image_size, crop_size):
        self.image_height, self.image_width = image_size
        self.crop_height, self.crop_width = crop_size

    def __call__(self, image):
        return image[(self.image_height - self.crop_height) // 2 : (self.image_height + self.crop_height) // 2,
                        (self.image_width - self.crop_width) // 2 : (self.image_width + self.crop_width) // 2, :]

class FlyingChairs(torch.utils.data.Dataset):
    def __init__(self, args, is_cropped, root):
        self.args = args
        self.is_cropped = is_cropped
        self.crop_size = args.crop_size
        self.render_size = args.inference_size

        images = sorted(glob(join(root, '*.ppm')))
        self.flow_list = sorted(glob(join(root, '*.flo')))
        assert (len(images)//2 == len(self.flow_list))
        self.image_list = []
        for idx in range(len(self.flow_list)):
            left_image = images[2*idx]
            right_image = images[2*idx + 1]
            self.image_list.append([left_image, right_image])
        assert (len(self.image_list) == len(self.flow_list))
        self.size = len(self.image_list)

        self.frame_size = utils.frame_utils.read_gen(self.image_list[0][0]).shape
        if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0] % 64) or (self.frame_size[1] % 64):
            self.render_size[0] = ((self.frame_size[0]) // 64) * 64
            self.render_size[1] = ((self.frame_size[1]) // 64) * 64
        args.inference_size = self.render_size

    def __getitem__(self, index):
        index = index % self.size
        left_image = utils.frame_utils.read_gen(self.image_list[index][0])
        right_image = utils.frame_utils.read_gen(self.image_list[index][1])
        flow = utils.frame_utils.read_gen(self.flow_list[index])

        image_size = left_image.shape[:2]
        if self.is_cropped:
            cropper = StaticRandomCrop(image_size, self.crop_size)
        else:
            cropper = StaticCenterCrop(image_size, self.render_size)

        left_image = cropper(left_image)
        right_image = cropper(right_image)
        flow = cropper(flow)

        left_image = left_image.transpose(2, 0, 1)
        right_image = right_image.transpose(2, 0, 1)
        flow = flow.transpose(2, 0, 1)

        left_image = torch.from_numpy(left_image.astype(np.float32))
        right_image = torch.from_numpy(right_image.astype(np.float32))
        flow = torch.from_numpy(flow.astype(np.float32))
        return left_image, right_image, flow

    def __len__(self):
        return self.size
