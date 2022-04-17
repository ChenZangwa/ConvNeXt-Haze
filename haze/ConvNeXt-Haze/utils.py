import os
import sys
import json
import pickle
import random
import math
import cv2
import torch
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F

class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=5, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重. 当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.255
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            print(" --- Focal_loss alpha = {}, 将对每一类权重进行精细化赋值 --- ".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]        [B*N个标签(假设框中有目标)]，[B个标签]
        :return:
        """

        # 固定类别维度，其余合并(总检测框数或总批次数)，preds.size(-1)是最后一个维度
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)

        # 使用log_softmax解决溢出问题，方便交叉熵计算而不用考虑值域
        preds_logsoft = F.log_softmax(preds, dim=1)

        # log_softmax是softmax+log运算，那再exp就算回去了变成softmax
        preds_softmax = torch.exp(preds_logsoft)

        # 这部分实现nll_loss ( crossentropy = log_softmax + nll)
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))

        self.alpha = self.alpha.gather(0, labels.view(-1))

        # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        # torch.mul 矩阵对应位置相乘，大小一致
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)

        # torch.t()求转置
        loss = torch.mul(self.alpha, loss.t())
        # print(loss.size()) [1,5]

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()

        return loss


def read_split_data(root: str, val_rate: float = 0.3):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "not find data for train."
    assert len(val_images_path) > 0, "not find data for eval"

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list

class LabelSmoothing(torch.nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # 此处的self.smoothing即我们的epsilon平滑参数。

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class FocalLossWithSmoothing(nn.Module):
    def __init__(
            self,
            num_classes: int,
            gamma: int = 1,
            lb_smooth: float = 0.15,
            size_average: bool = True,
            ignore_index: int = None,
            alpha: float = None):
        """
        :param gamma:
        :param lb_smooth:
        :param ignore_index:
        :param size_average:
        :param alpha:
        """
        super(FocalLossWithSmoothing, self).__init__()
        self._num_classes = num_classes
        self._gamma = gamma
        self._lb_smooth = lb_smooth
        self._size_average = size_average
        self._ignore_index = ignore_index
        self._log_softmax = nn.LogSoftmax(dim=1)
        self._alpha = alpha

        if self._num_classes <= 1:
            raise ValueError('The number of classes must be 2 or higher')
        if self._gamma < 0:
            raise ValueError('Gamma must be 0 or higher')
        if self._alpha is not None:
            if self._alpha <= 0 or self._alpha >= 1:
                raise ValueError('Alpha must be 0 <= alpha <= 1')

    def forward(self, logits, label):
        """
        :param logits: (batch_size, class, height, width)
        :param label:
        :return:
        """
        logits = logits.float()
        difficulty_level = self._estimate_difficulty_level(logits, label)

        with torch.no_grad():
            label = label.clone().detach()
            if self._ignore_index is not None:
                ignore = label.eq(self._ignore_index)
                label[ignore] = 0
            lb_pos, lb_neg = 1. - self._lb_smooth, self._lb_smooth / (self._num_classes - 1)
            lb_one_hot = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()
        logs = self._log_softmax(logits)
        loss = -torch.sum(difficulty_level * logs * lb_one_hot, dim=1)
        if self._ignore_index is not None:
            loss[ignore] = 0
        return loss.mean()

    def _estimate_difficulty_level(self, logits, label):
        """
        :param logits:
        :param label:
        :return:
        """
        one_hot_key = torch.nn.functional.one_hot(label, num_classes=self._num_classes)
        if len(one_hot_key.shape) == 4:
            one_hot_key = one_hot_key.permute(0, 3, 1, 2)
        if one_hot_key.device != logits.device:
            one_hot_key = one_hot_key.to(logits.device)
        pt = one_hot_key * F.softmax(logits)
        difficulty_level = torch.pow(1 - pt, self._gamma)
        return difficulty_level

def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler):
    model.train()
    # loss_function1 = torch.nn.CrossEntropyLoss()
    # loss_function2 = LabelSmoothing(smoothing=0.15)
    # loss_function3 = focal_loss()
    loss_function4 = FocalLossWithSmoothing(num_classes=5)
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()


    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels  = data

        # images = cutblur(images)

        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        # loss = loss_function1(pred, labels.to(device))
        # loss = loss_function2(pred, labels.to(device))
        # loss = loss_function3(pred, labels.to(device))
        loss = loss_function4(pred, labels.to(device))
        # pred:[32,5] lable:[32]

        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            optimizer.param_groups[0]["lr"]
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        # update lr
        lr_scheduler.step()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()
    true_label_list = []
    pred_label_list = []
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels  = data
        sample_num += images.shape[0]

        pred = model(images.to(device))


        pred_classes = torch.max(pred, dim=1)[1]
        true_label_list.append(labels.cpu().detach().numpy())
        pred_label_list.append(pred_classes.cpu().detach().numpy())

        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num
        )
    y_true = np.concatenate(true_label_list)
    y_pred = np.concatenate(pred_label_list)
    precision, recall, f1 = precision_recall_fscore_support(y_true, y_pred, average='macro')[:-1]
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num ,precision, recall, f1


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    # 记录optimize要训练的权重参数
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    # 记录对应的权重名称
    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())
