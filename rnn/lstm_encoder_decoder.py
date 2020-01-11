import os
import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils import data
import torch.nn.utils.rnn as rnn

import matplotlib.pyplot as plt
import time
import wsj
import csv
import phoneme_list
from ctcdecode import CTCBeamDecoder

# Global
CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")
os.environ["WSJ_PATH"] = "."
PHONEME_MAP = [' '] + phoneme_list.PHONEME_MAP


class ASRDataset(data.Dataset):
    def __init__(self, X, Y):
        super(ASRDataset, self).__init__()
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self,index):
        return self.X[index], self.Y[index]

class ASREncoder(nn.Module):
    def __init__(self, embed_size, hidden_size, nlayers):
        super(ASREncoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.nlayers=nlayers
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size,
                            num_layers=nlayers, bidirectional=True, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=hidden_size*2, out_features=1024),
            nn.Linear(in_features=1024, out_features=2048),
            nn.Linear(in_features=2048, out_features=phoneme_list.N_PHONEMES+1)
        )
        self.linear = nn.Linear(in_features=hidden_size*2, out_features=phoneme_list.N_PHONEMES+1)

    def forward(self, X, lengths):
        packed_X = rnn.pack_padded_sequence(X, lengths, enforce_sorted=False)
        packed_out = self.lstm(packed_X)[0]
        out, out_lens = rnn.pad_packed_sequence(packed_out)
        # Log softmax after output layer is required for use in `nn.CTCLoss`.
        out = self.classifier(out).log_softmax(2)
        return out, out_lens

def collate(data):
    inputs, targets = zip(*data)

    # Add 1 to all labels
    targets = [torch.from_numpy(l+1) for l in targets]
    inputs = [torch.from_numpy(x) for x in inputs]
    inputs_len = torch.tensor([len(x) for x in inputs])
    targets_len = torch.tensor([len(l) for l in targets])

    # pad data
    inputs = rnn.pad_sequence(inputs)
    targets = rnn.pad_sequence(targets, batch_first=True)

    return inputs, targets, inputs_len, targets_len

def train(train_loader, test_loader, encoder, epochs):
    criterion = nn.CTCLoss()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)

    for epoch in range(epochs):
        encoder.train()
        total_loss = 0.0
        for i, data in enumerate(train_loader):
            encoder.train()
            X, Y, X_len, Y_len = data
            X, Y, X_len, Y_len = X.to(DEVICE), Y.to(DEVICE), X_len.to(DEVICE), Y_len.to(DEVICE)
            out, out_len = encoder(X, X_len)
            loss = criterion(out, Y, out_len, Y_len)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print("Batch: {}, Loss: {}".format(i, loss.item()))

            torch.cuda.empty_cache()
            del X
            del Y
            del X_len
            del Y_len
            del loss
        print('Epoch', epoch + 14, 'Loss', total_loss)
        torch.save(encoder.state_dict(), "encoder_"+str(epoch+14))
        test(test_loader, encoder)

def test(test_loader, encoder):
    encoder.eval()
    decoder = CTCBeamDecoder(['$'] * (phoneme_list.N_PHONEMES+1), beam_width=100,
                                    log_probs_input=True, num_processes=os.cpu_count())

    with open("result.csv", "w") as f:
        print("Id,Predicted", file=f)
        for i, data in enumerate(test_loader):
            X, _, X_len, _ = data
            X, X_len = X.to(DEVICE), X_len.to(DEVICE)
            out, out_len = encoder(X, X_len)
            test_Y, _, _, test_Y_lens = decoder.decode(out.transpose(0, 1), out_len)
            for j in range(len(test_Y)):
                best_seq = test_Y[j, 0, :test_Y_lens[j, 0]]
                best_pron = ''.join(PHONEME_MAP[k] for k in best_seq)
                print(str(i*64 + j)+","+best_pron, file=f)


if __name__ == "__main__":
    loader = wsj.WSJ()
    trainX, trainY = loader.train
    # valX, valY = loader.dev
    testX, _ = loader.test

    train_dataset = ASRDataset(trainX, trainY)
    test_dataset = ASRDataset(testX, [np.array([[0]])]*len(testX))
    train_loader = data.DataLoader(train_dataset, shuffle=True, batch_size=64, collate_fn=collate)
    test_loader = data.DataLoader(test_dataset, shuffle=False, batch_size=64, collate_fn=collate)

    encoder = ASREncoder(embed_size=40, hidden_size=256, nlayers=4)
    encoder.load_state_dict(torch.load('encoder_13'))
    encoder.to(DEVICE)

    train(train_loader, test_loader, encoder, 20)
