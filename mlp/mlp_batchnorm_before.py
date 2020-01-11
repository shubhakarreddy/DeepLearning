import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils import data

import matplotlib.pyplot as plt
import time
import wsj
import bisect
import csv
import os

# Global
CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")
K = 12
FEATURES = 40
OUTPUT_SIZE = 138

os.environ["WSJ_PATH"] = "/home/ubuntu/11785/11-785hw1p2-f19"

class ASRDataset(data.Dataset):
    def __init__(self, X, Y):
        super(ASRDataset, self).__init__()
        self.X = X
        self.Y = Y
        count, self.frame_pos = 0, []
        for i in range(len(X)):
            count += len(X[i])
            self.frame_pos += [count]

    def __len__(self):
        return self.frame_pos[-1]

    def __getitem__(self,index):
        utt_num = bisect.bisect_right(self.frame_pos, index)
        if utt_num != 0:
            index -= self.frame_pos[utt_num-1]

        # pad
        utterence = self.X[utt_num]
        pad = np.zeros((K, FEATURES))
        utterence = np.concatenate((pad, utterence, pad), axis=0)

        X = utterence[index:index+(2*K+1)].reshape(-1)

        Y = []
        if len(self.Y) != 0:
          Y = self.Y[utt_num][index]

        return torch.from_numpy(X).type(torch.FloatTensor), Y

def init_randn(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0,1)

def init_xavier(m):
    if type(m) == nn.Linear:
        fan_in = m.weight.size()[1]
        fan_out = m.weight.size()[0]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        m.weight.data.normal_(0,std)

class ASRMLP(nn.Module):
    def __init__(self, input_size, output_size, hiddens, num_bn_layers=0, dropout=0.0):
        super(ASRMLP, self).__init__()
        layers_size = [input_size] + hiddens + [output_size]
        layers = [nn.Linear(layers_size[0], layers_size[1])]
        for i in range(1, len(layers_size)-1):
            layers += [nn.BatchNorm1d(layers_size[i])]
            layers += [nn.ReLU()]
            layers += [nn.Linear(layers_size[i], layers_size[i+1])]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train(epochs, train_dataloader, val_dataloader, test_dataloader, weight_init_fn, lr, *args):
    # *args = input_size, output_size, hiddens, weight_init_fn, lr=1e-3, num_bn_layers=0
    model = ASRMLP(*args)
    model.apply(weight_init_fn)
    model.train()
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_loss = 0.0
    start_time = time.time()
    pred = []
    for epoch in range(epochs):
        start_epoch = time.time()
        print("starting epoch:", epoch)
        for i, (X, Y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            X = X.to(DEVICE)
            Y = Y.to(DEVICE)

            output = model.forward(X)
            loss = criterion(output, Y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if i % 5000 == 0:
                print("batch:", i, "loss:", loss.item())

        if epoch == 10:
            for param_group in optimizer.param_groups:
                    param_group['lr'] = lr/10

        end_epoch = time.time()
        print('Training Loss:', loss.item(), "Time taken:", end_epoch-start_epoch)
        validate(model, val_dataloader, criterion)

        if epoch % 3 == 0:
            torch.save(model.state_dict(), "torch_model")
            test(model, test_dataloader, epoch)

    end_time = time.time()
    total_loss /= len(train_dataloader)

    print('Training Loss: ', total_loss, "Time taken:", end_time-start_time)
    return model


def validate(model, val_dataloader, criterion):
    with torch.no_grad():
        model.eval()
        model.to(DEVICE)

        total_loss = 0.0
        total_pred_batch = 0.0
        correct_pred_batch = 0.0
        total_pred = []

        for _, (X, Y) in enumerate(val_dataloader):
            X = X.to(DEVICE)
            Y = Y.to(DEVICE)

            output = model(X)

            _, pred = torch.max(output.data, 1)
            total_pred_batch += X.size(0)
            correct_pred_batch += (pred == Y).sum().item()

            loss = criterion(output, Y).detach()
            total_loss += loss.item()

        total_loss /= len(val_dataloader)
        acc = (correct_pred_batch/total_pred_batch)*100.0
        print('Validation Loss: ', total_loss)
        print('Validation Accuracy: ', acc, '%')


def test(model, test_dataloader, epoch):
    with torch.no_grad():
        model.eval()
        model.to(DEVICE)

        total_pred = torch.LongTensor([]).to(DEVICE)

        for _, (X, Y) in enumerate(test_dataloader):
            X = X.to(DEVICE)

            output = model(X)
            _, pred = torch.max(output.data, 1)
            total_pred = torch.cat([total_pred, pred])

        PrintOutput("test_output_{}.csv".format(epoch), total_pred)


def PrintOutput(csv_path, pred):
    with open(csv_path, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerow(["id","label"])
        for i, val in enumerate(pred):
            writer.writerow([i,val.item()])

if __name__ == "__main__":
    loader = wsj.WSJ()
    trainX, trainY = loader.train
    print(len(trainX))
    # valX, valY = loader.dev
    # testX, testY = loader.test
    #
    # train_dataset = ASRDataset(trainX, trainY)
    #
    # train_loader_args = dict(shuffle=True, batch_size=1024, num_workers=8, pin_memory=True) if CUDA\
    #                 else dict(shuffle=True, batch_size=16)
    # train_dataloader = data.DataLoader(train_dataset, **train_loader_args)
    #
    # val_dataset = ASRDataset(valX, valY)
    # val_loader_args = dict(shuffle=False, batch_size=1024, num_workers=8, pin_memory=True) if CUDA\
    #                 else dict(shuffle=False, batch_size=16)
    # val_dataloader = data.DataLoader(val_dataset, **val_loader_args)
    #
    # # Use at the end
    # test_dataset = ASRDataset(testX, [])
    # test_loader_args = dict(shuffle=False, batch_size=1024, num_workers=8, pin_memory=True) if CUDA\
    #                 else dict(shuffle=False, batch_size=16)
    # test_dataloader = data.DataLoader(test_dataset, **test_loader_args)
    #
    # # hyper params
    # epochs = 20
    # input_size = FEATURES*(2*K+1)
    # hiddens = [2048, 1024, 1024, 1024, 512]
    # model = train(epochs, train_dataloader, val_dataloader, test_dataloader, init_randn, 1e-3, input_size, OUTPUT_SIZE, hiddens)
    # test(model, test_dataloader, 20)
