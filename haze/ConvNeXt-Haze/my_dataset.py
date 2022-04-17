from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

class Cutblur(object):
    def __init__(self):
        self.name = 'CutGaussianBlur'

    @staticmethod
    def __call__(img, prob=0.5, alpha=0.95):

        im2 = img.copy()
        im2 = im2.filter(ImageFilter.GaussianBlur(radius=5))

        if alpha <= 0 or np.random.rand(1) >= prob:
            return img

        cut_ratio = np.random.randn() * 0.01 + alpha

        def get_params(img):
            if type(img) == np.ndarray:
                img_h, img_w, img_c = img.shape
            else:
                img_h, img_w = img.size
                img_c = len(img.getbands())

            return img_h, img_w
        h, w = get_params(img)
        ch, cw = np.int32(h * cut_ratio), np.int32(w * cut_ratio)
        # print(ch,cw,h,w)
        cy = np.random.randint(0, h - ch + 1)
        cx = np.random.randint(0, w - cw + 1)

        # apply CutBlur to inside or outside
        im2 = cv2.cvtColor(np.array(im2), cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        if np.random.random() > 0.5:
            im2[cy:cy + ch, cx:cx + cw] = img[cy:cy + ch, cx:cx + cw]
        else:
            im2_aug = img.copy()
            im2_aug[cy:cy + ch, cx:cx + cw] = im2[cy:cy + ch, cx:cx + cw]
            im2 = im2_aug
        im2 = Image.fromarray(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))
        return im2


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform
        self.cutblur = Cutblur()

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        img = img.convert('RGB')
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]
        img = self.cutblur(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels= tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
