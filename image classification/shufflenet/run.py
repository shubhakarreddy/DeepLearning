import os
import numpy as np
from PIL import Image

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import model
import time
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImageDataset(Dataset):
    def __init__(self, file_list, target_list):
        self.file_list = file_list
        self.target_list = target_list
        self.n_class = len(list(set(target_list)))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img = Image.open(self.file_list[index])
        img = torchvision.transforms.ToTensor()(img)
        label = self.target_list[index]
        return img, label

def parse_data(datadir):
    img_list = []
    ID_list = []
    for root, directories, filenames in os.walk(datadir):
        for filename in filenames:
            if filename.endswith('.jpg'):
                filei = os.path.join(root, filename)
                img_list.append(filei)
                ID_list.append(root.split('/')[-1])

    # construct a dictionary, where key and value correspond to ID and target
    uniqueID_list = list(set(ID_list))
    class_n = len(uniqueID_list)
    target_dict = pickle.load(open("../target_dict.pickle", 'rb'))
    target_dict_rev = pickle.load(open("../target_dict_rev.pickle", 'rb'))
    # target_dict = dict(zip(uniqueID_list, range(class_n)))
    # target_dict_rev = dict(zip(range(class_n), uniqueID_list))
    # pickle.dump(target_dict, open("target_dict.pickle", 'wb'))
    # pickle.dump(target_dict_rev, open("target_dict_rev.pickle", 'wb'))
    label_list = [target_dict[ID_key] for ID_key in ID_list]
    print('{}\t\t{}\n{}\t\t{}'.format('#Images', '#Labels', len(img_list), len(set(label_list))))
    return img_list, label_list, class_n, target_dict_rev

def train(model, data_loader, val_loader, test_loader, label_to_org, optimizer, criterion, numEpochs, task='Classification'):
    model.train()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)

    for epoch in range(numEpochs):
        avg_loss = 0.0
        start = time.time()
        for batch_num, (feats, labels) in enumerate(data_loader):
            feats, labels = feats.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(feats)[1]

            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

            if batch_num % 1000 == 999:
                end = time.time()
                print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}\tTime: {}'.format(epoch+1, batch_num+1, avg_loss/1000, end-start))
                avg_loss = 0.0
                start = time.time()

            torch.cuda.empty_cache()
            del feats
            del labels
            del loss

        scheduler.step()
        if task == 'Classification':
            start = time.time()
            val_loss, val_acc = test_classify(model, val_loader)
            test_pred = predict_classify(model, test_loader)
            PrintData(test_pred, "test_classification_output_" + str(epoch+1) + ".csv", label_to_org)
            # train_loss, train_acc = test_classify(model, data_loader)
            end = time.time()
            print('########################\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}\tTime: {:.4f}\t######################'.
                  format(val_loss, val_acc, end-start))
            torch.save(model.state_dict(), "model_"+str(epoch+1))
        else:
            test_verify(model, test_loader)

def PrintData(test_pred, filename, target_dict_rev):
    indices = range(5000, 9600)
    with open(filename, "w") as f:
        print("id"+","+"label", file=f)
        for index, c in zip(indices, test_pred):
            print(str(index)+","+str(label_to_org[c.item()]), file=f)

def test_classify(model, test_loader):
    model.eval()
    test_loss = []
    accuracy = 0
    total = 0

    for batch_num, (feats, labels) in enumerate(test_loader):
        feats, labels = feats.to(device), labels.to(device)
        outputs = model(feats)[1]

        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)

        loss = criterion(outputs, labels.long())

        accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        test_loss.extend([loss.item()]*feats.size()[0])
        del feats
        del labels

    model.train()
    return np.mean(test_loss), accuracy/total

def predict_classify(model, test_loader):
    model.eval()
    pred_labels_total = torch.tensor([]).type(torch.LongTensor).to(device)

    for batch_num, (feats, labels) in enumerate(test_loader):
        feats, labels = feats.to(device), labels.to(device)
        outputs = model(feats)[1]

        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)
        pred_labels_total = torch.cat((pred_labels_total, pred_labels))

        del feats
        del labels

    model.train()
    return pred_labels_total

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)

if __name__ == "__main__":
    img_list, label_list, class_n, label_to_org = parse_data('../data/train_data/medium')
    trainset = ImageDataset(img_list, label_list)

    img_list_val, label_list_val, class_n_val, _ = parse_data('../data/validation_classification/medium')
    valset = ImageDataset(img_list_val, label_list_val)


    img_list_test = ["../data/test_classification/medium/" + str(x) + ".jpg" for x in range(5000, 9600)]
    label_list_test = [1]*len(img_list_test)
    testset = ImageDataset(img_list_test, label_list_test)
    dataloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=1, drop_last=False)
    valloader = DataLoader(valset, batch_size=128, shuffle=True, num_workers=1, drop_last=False)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=1, drop_last=False)

    output_channels = {1: 24, 2: 240, 3: 480, 4: 960}
    # repeats = {2: 3, 3: 5, 4: 2}
    repeats = {2: 3, 3: 7, 4: 3}
    shufflenet = model.ShuffleNet(3, output_channels, 3, repeats, class_n)
    # shufflenet.load_state_dict(torch.load('model_13', map_location=torch.device('cpu')))
    shufflenet.to(device)
    shufflenet.apply(init_weights)

    learningRate = 1e-2
    # weightDecay = 5e-7
    numEpochs = 20

    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(shufflenet.parameters(), lr=learningRate, weight_decay=weightDecay, momentum=0.97)
    optimizer = torch.optim.Adam(shufflenet.parameters(), lr=learningRate)

    train(shufflenet, dataloader, valloader, testloader, label_to_org, optimizer, criterion, numEpochs)
