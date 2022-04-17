import os
import numpy as np
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import pickle

"""
在网络训练前先运行该函数获得数据的均值和标准差
"""


class Dataloader():
    def __init__(self, dataroot):
        self.dataroot = dataroot
        self.dirs = ['haze_data']

        self.means = [0, 0, 0]
        self.std = [0, 0, 0]

        self.transform = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.ToTensor()  # 数据值从[0,255]范围转为[0,1]，相当于除以255操作
                                             ])

        # 因为这里使用的是ImageFolder，按文件夹给数据分类，一个文件夹为一类，label会自动标注好
        self.dataset = {x: ImageFolder(os.path.join(dataroot, x), self.transform) for x in self.dirs}

    def get_mean_std(self):
        """
        计算数据集的均值和标准差
        """
        num_imgs = len(self.dataset['haze_data'])
        for data in self.dataset['haze_data']:
            img = data[0]
            for i in range(3):
                # 计算每一个通道的均值和标准差
                self.means[i] += img[i, :, :].mean()
                self.std[i] += img[i, :, :].std()

        self.means = np.asarray(self.means) / num_imgs
        self.std = np.asarray(self.std) / num_imgs

        print("{}: normMean = {}".format(type, self.means))
        print("{}: normstd = {}".format(type, self.std))

        # # 将得到的均值和标准差写到文件中，之后就能够从中读取
        # with open(mean_std_path, 'wb') as f:
        #     pickle.dump(self.means, f)
        #     pickle.dump(self.stdevs, f)
        #     print('pickle done')


if __name__ == '__main__':
    dataroot = '/home/chenhao/ZangChen/pytorch_learn'
    dataloader = Dataloader(dataroot)
    dataloader.get_mean_std()