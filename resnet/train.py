import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import numpy as np
from model import resnet101
import time
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import LambdaLR
def cnn_paras_count(net):
    """cnn参数量统计, 使用方式cnn_paras_count(net)"""
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in net.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    return total_params, total_trainable_params

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "./"))
    image_path = os.path.join(data_root, "haze_data_alex")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    haze_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in haze_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 128
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    
    net = resnet101()
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model_weight_path = "./resnet101-pre.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    # #todo:自定义载入
    # pre_weight = torch.load(model_weight_path, map_location=device)
    # model_weight = net.state_dict()
    # excludekey=["conv1.weight"]
    # select_weight = {k:v for k,v in pre_weight.items() if k in  model_weight and not k in excludekey}
    # net.load_state_dict(select_weight)

    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 5)

    net = nn.DataParallel(net)
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.00001)
    # scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epoch + 1))
    epochs = 200
    best_acc = 0.0
    save_path = './resNet101.pth'
    train_steps = len(train_loader)
    best_epoch = 0

    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # scheduler.step()
        print("第%d个epoch的学习率：%f" % (epoch+1, optimizer.param_groups[0]['lr']))
        # validate
        net.eval()
        start_time = time.time()
        total_test_loss = 0
        true_label_list = []
        pred_label_list = []
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                true_label_list.append(val_labels.cpu().detach().numpy())
                pred_label_list.append(predict_y.cpu().detach().numpy())
                loss = loss_function(outputs, val_labels.to(device))
                total_test_loss += loss.item()
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)
        y_true = np.concatenate(true_label_list)
        y_pred = np.concatenate(pred_label_list)
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1 = precision_recall_fscore_support(y_true, y_pred, average='macro')[:-1]
        print("accuracy: ", accuracy)
        print("precision: ", precision)
        print("recall: ", recall)
        print("f1: ", f1)
        print("average_loss: ", total_test_loss / val_num)
        # print("网络参数: ", cnn_paras_count(net)[0])
        print('网络参数大小(MB): %.3fMB'% (cnn_paras_count(net)[0]*4/1024/1024))
        val_accurate = acc / val_num
        end_time = time.time()
        print("验证183张图片运行总时间为：{} ".format((end_time - start_time)))
        # print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
        #       (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_epoch = epoch + 1
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f  best_accuracy: %.3f  best_epoch: %.0d' %
              (epoch + 1, running_loss / train_steps, val_accurate, best_acc, best_epoch))
    print('Finished Training')



if __name__ == '__main__':
    main()
