import os
import sys
import copy
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def train(model, values, train_dataloaders, train_sizes, criterion, optimizer, epochs):
    time_stamp = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1000000000
    for epoch in range(epochs):
        print('-' * 100)
        print('epoch {}/{}'.format(epoch + 1, epochs))
        for mode in ['train/', 'val/']:
            if mode == 'train/':
                model.train()
            else:
                model.eval()
            running_loss = 0
            for inputs, indices in train_dataloaders[mode]:
                inputs = inputs.to(device)
                labels = torch.zeros(len(indices)).to(device)
                for index in range(len(indices)):
                    labels[index] = float(values[indices[index]])
                optimizer.zero_grad()
                with torch.set_grad_enabled(mode == 'train/'):
                    outputs = torch.squeeze(model(inputs))
                    loss = criterion(outputs, labels)
                    if mode == 'train/':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / train_sizes[mode]
            print('{} loss: {:.4f}'.format(mode.split('/')[0], epoch_loss))
            if mode == 'val/' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
    time_elapsed = time.time() - time_stamp
    print('-' * 100)
    datetime_now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    print('{}\ntraining completed in {:.0f}m {:.0f}s'.format(datetime_now, time_elapsed // 60, time_elapsed % 60))
    print('best val loss: {:.4f}'.format(best_loss))
    model.load_state_dict(best_model_wts)
    return model


def apply(model, values, num_values, apply_dataloader, apply_size):
    model.eval()
    time_stamp = time.time()
    num_samples = torch.zeros(num_values)
    preds = torch.zeros([num_values, int(apply_size/num_values)])
    for inputs, indices in apply_dataloader:
        inputs = inputs.to(device)
        labels = torch.zeros(len(indices)).to(device)
        for index in range(len(indices)):
            labels[index] = float(values[indices[index]])
        with torch.no_grad():
            outputs = torch.squeeze(model(inputs))
        for sample in range(outputs.size(0)):
            preds[indices.data[sample], int(num_samples[indices.data[sample]])] = outputs.data[sample]
            num_samples[indices.data[sample]] += 1
    preds_mean = torch.mean(preds, dim=1)
    preds_std = torch.std(preds, dim=1)
    for value in range(num_values):
        v = torch.zeros(1)
        v[0] = float(values[value])
    time_elapsed = time.time() - time_stamp
    print('testing completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('preds_mean =', preds_mean, '\npreds_std =', preds_std)
    return preds_mean, preds_std


if __name__ == '__main__':
    T_table = ['2.200', '2.220', '2.240', '2.260', '2.280',
               '2.300', '2.320', '2.340', '2.360', '2.380', '2.400']

    root = './data_Ising120/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_values = len(T_table)
    average_mean = torch.zeros(num_values)
    average_std = torch.zeros(num_values)

    train_datasets = {x: datasets.ImageFolder(os.path.join(root, x), transforms.ToTensor()) for x in ['train/', 'val/']}
    train_sizes = {x: len(train_datasets[x]) for x in ['train/', 'val/']}
    train_dataloaders = {x: DataLoader(train_datasets[x], batch_size=500, shuffle=True, num_workers=4, pin_memory=True) for x in ['train/', 'val/']}

    apply_dataset = datasets.ImageFolder(os.path.join(root, 'test/'), transforms.ToTensor())
    apply_size = len(apply_dataset)
    apply_dataloader = DataLoader(apply_dataset, batch_size=500, shuffle=True, num_workers=4, pin_memory=True)

    num_average = 10
    for j in range(num_average):
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 1)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model = train(model, T_table, train_dataloaders, train_sizes, nn.MSELoss(), optimizer, epochs=20)
        preds_mean, preds_std = apply(model, T_table, num_values, apply_dataloader, apply_size)
        average_mean += preds_mean
        average_std += preds_std
    average_mean /= num_average
    average_std /= num_average

    print('-' * 100)
    print('mean =', average_mean, '\nstd =', average_std)


