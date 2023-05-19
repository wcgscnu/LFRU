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


def train(model, train_dataloaders, train_sizes, criterion, optimizer, epochs):
    time_stamp = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0
    for epoch in range(epochs):
        print('-' * 100)
        print('epoch {}/{}'.format(epoch + 1, epochs))
        for mode in ['train/', 'val/']:
            if mode == 'train/':
                model.train()
            else:
                model.eval()
            running_loss = 0
            running_corrects = 0
            for inputs, labels in train_dataloaders[mode]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(mode == 'train/'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if mode == 'train/':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / train_sizes[mode]
            epoch_acc = running_corrects.double() / train_sizes[mode]
            print('{} loss: {:.4f}\tacc: {:.4f}'.format(mode.split('/')[0], epoch_loss, epoch_acc))
            if mode == 'val/' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    time_elapsed = time.time() - time_stamp
    print('-' * 100)
    datetime_now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    print('{}\ntraining completed in {:.0f}m {:.0f}s'.format(datetime_now, time_elapsed // 60, time_elapsed % 60))
    print('best val acc: {:.4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model


def apply(model, apply_dataloader, apply_size):
    model.eval()
    time_stamp = time.time()
    running_corrects = 0
    for inputs, labels in apply_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
    total_acc = running_corrects.double() / apply_size
    time_elapsed = time.time() - time_stamp
    print('testing completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('total test acc: {:.4f}'.format(total_acc))
    return total_acc


if __name__ == '__main__':

    root = './data_Ising120/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    average_apply_acc = 0

    train_datasets = {x: datasets.ImageFolder(os.path.join(root, x), transforms.ToTensor()) for x in ['train/', 'val/']}
    train_sizes = {x: len(train_datasets[x]) for x in ['train/', 'val/']}
    train_dataloaders = {x: DataLoader(train_datasets[x], batch_size=500, shuffle=True, num_workers=4, pin_memory=True) for x in ['train/', 'val/']}

    apply_dataset = datasets.ImageFolder(os.path.join(root, 'test/'), transforms.ToTensor())
    apply_size = len(apply_dataset)
    apply_dataloader = DataLoader(apply_dataset, batch_size=500, shuffle=True, num_workers=4, pin_memory=True)

    num_average = 10
    for j in range(num_average):
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model = train(model, train_dataloaders, train_sizes, nn.CrossEntropyLoss(), optimizer, epochs=20)
        average_apply_acc += apply(model, apply_dataloader, apply_size)
    average_apply_acc /= num_average

    print('-' * 100)
    print('average_test_acc = {:.4f}'.format(average_apply_acc))

