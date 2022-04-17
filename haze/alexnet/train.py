import os
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import time
from model import AlexNet
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "./"))  # get data root path
    image_path = os.path.join(data_root, "haze_data_alex")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    haze_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in haze_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 64
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=16, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()
    #
    # def imshow(img):
    #     img = img / 2 + 0.5  # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #
    # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(test_image))

    net = AlexNet(num_classes=5, init_weights=True)

    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    # pata = list(net.parameters())
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    epochs = 200
    save_path = './AlexNet.pth'
    best_acc = 0.0
    train_steps = len(train_loader)
    best_epoch = 0
    start_time = time.time()
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        # 长度为K类的数组
        # acc_per = [0]*5
        # data_quantity=[0]*5
        # acc_i=[0]*5
        # precision=[0]*5
        # recall=[0]*5
        total_test_loss = 0
        true_label_list = []
        pred_label_list = []
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # predict_y = torch.max(outputs, dim=1)[1]
                predict_y = outputs.argmax(dim=1)
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                true_label_list.append(val_labels.cpu().detach().numpy())
                pred_label_list.append(predict_y.cpu().detach().numpy())
                loss = loss_function(outputs, val_labels.to(device))
                total_test_loss += loss.item()
        y_true = np.concatenate(true_label_list)
        y_pred = np.concatenate(pred_label_list)
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1 = precision_recall_fscore_support(y_true, y_pred, average='macro')[:-1]
        print("accuracy: ", accuracy)
        print("precision: ", precision)
        print("recall: ", recall)
        print("f1: ", f1)
        print("average_loss: ", total_test_loss / val_num)


        val_accurate = acc / val_num

        if val_accurate > best_acc:
            best_epoch = epoch + 1
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f  best_accuracy: %.3f  best_epoch: %.0d' %
              (epoch + 1, running_loss / train_steps, val_accurate, best_acc, best_epoch))



    print('Finished Training')
    end_time = time.time()
    print("程序的运行总时间为：{}".format((end_time - start_time)))


if __name__ == '__main__':
    main()
