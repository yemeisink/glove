import os
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

batch_size = 32
learning_rate = 0.001
num_epochs = 25
train_data = []
train_labels = []

# read bare
train_new_path = os.path.join(os.getcwd(), "new_train")
one_path = os.path.join(train_new_path, "bare")
csv_files = [os.path.join(one_path, "2023_0912_ba.csv"),
             os.path.join(one_path, "2023_0912_bare.csv"),
             os.path.join(one_path, "20230912_bare.csv"),
             os.path.join(one_path, "20230912_ba.csv"),]

for csv_file in csv_files:
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) > 1:
                row_data = list(map(int, row[:-1]))
                train_data.append(row_data)
                train_labels.append(0)

# read bottom disk
train_new_path = os.path.join(os.getcwd(), "new_train")
one_path = os.path.join(train_new_path, "bottom")
csv_files = [os.path.join(one_path, "2023_0912_bo.csv"),
             os.path.join(one_path, "2023_0912_bottom.csv"),
             os.path.join(one_path, "20230912_bottom.csv"),
             os.path.join(one_path, "20230912_bo.csv"),]

for csv_file in csv_files:
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) > 1:
                row_data = list(map(int, row[:-1]))
                train_data.append(row_data)
                train_labels.append(1)

# read mid disk
train_new_path = os.path.join(os.getcwd(), "new_train")
one_path = os.path.join(train_new_path, "mid")
csv_files = [os.path.join(one_path, "2023_0912_mi.csv"),
             os.path.join(one_path, "2023_0912_mid.csv"),
             os.path.join(one_path, "20230912_mid.csv"),
             os.path.join(one_path, "20230912_mi.csv"),]

for csv_file in csv_files:
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) > 1:
                row_data = list(map(int, row[:-1]))
                train_data.append(row_data)
                train_labels.append(2)

# read top disk
train_new_path = os.path.join(os.getcwd(), "new_train")
one_path = os.path.join(train_new_path, "top")
csv_files = [os.path.join(one_path, "2023_0912_to.csv"),
             os.path.join(one_path, "2023_0912_top.csv"),
             os.path.join(one_path, "20230912_top.csv"),
             os.path.join(one_path, "20230912_to.csv"),]

for csv_file in csv_files:
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) > 1:
                row_data = list(map(int, row[:-1]))
                train_data.append(row_data)
                train_labels.append(3)

# Test data

'''
test_data = []
test_labels = []

# bare test
test_new_path = os.path.join(os.getcwd(), "new_test")
one_path = os.path.join(test_new_path, "bare")
csv_files = [os.path.join(one_path, "2023_0912_ba.csv"),
             os.path.join(one_path, "2023_0912_bare.csv"),]

for csv_file in csv_files:
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) > 1:
                row_data = list(map(int, row[:-1]))
                test_data.append(row_data)
                test_labels.append(0)


# bottom test
test_new_path = os.path.join(os.getcwd(), "new_test")
one_path = os.path.join(test_new_path, "bottom")
csv_files = [os.path.join(one_path, "2023_0912_bo.csv"),
             os.path.join(one_path, "2023_0912_bottom.csv"),]

for csv_file in csv_files:
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) > 1:
                row_data = list(map(int, row[:-1]))
                test_data.append(row_data)
                test_labels.append(1)

# mid test
test_new_path = os.path.join(os.getcwd(), "new_test")
one_path = os.path.join(test_new_path, "mid")
csv_files = [os.path.join(one_path, "2023_0912_mi.csv"),
             os.path.join(one_path, "2023_0912_mid.csv"),]

for csv_file in csv_files:
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) > 1:
                row_data = list(map(int, row[:-1]))
                test_data.append(row_data)
                test_labels.append(2)

# top test
test_new_path = os.path.join(os.getcwd(), "new_test")
one_path = os.path.join(test_new_path, "top")
csv_files = [os.path.join(one_path, "2023_0912_to.csv"),
             os.path.join(one_path, "2023_0912_top.csv"),]

for csv_file in csv_files:
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) > 1:
                row_data = list(map(int, row[:-1]))
                test_data.append(row_data)
                test_labels.append(3)
'''

# put data to numpy
train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

train_data_fold = np.array(train_data)
train_labels_fold = np.array(train_labels)

test_data_fold = np.array(test_data)
test_labels_fold = np.array(test_labels)

# tensor
train_data_fold = torch.Tensor(train_data_fold).unsqueeze(1)
train_labels_fold = torch.LongTensor(train_labels_fold)

test_data_fold = torch.Tensor(test_data_fold).unsqueeze(1)
test_labels_fold = torch.LongTensor(test_labels_fold)

'''创建迭代器dataloader'''
train_dataset = TensorDataset(train_data_fold, train_labels_fold)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(test_data_fold, test_labels_fold)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


class CNN1D(nn.Module):
    def __init__(self, num_classes):
        super(CNN1D, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=15, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(in_channels=15, out_channels=30, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(750, 240),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(240, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)

        return x

# Assuming input size (sequence length) is 92 and number of classes is 4

num_classes = 4

model = CNN1D(num_classes)

'''构建优化器'''
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

'''损失函数'''
criterion = nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''gpu训练'''
model = model.to(device)
criterion = criterion.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0

    for inputs, targets in train_loader:
        inputs , targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == targets).sum().item()

    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = correct / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.4f}")

    model.eval()  # 设置为评估模式
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()

        # 打印测试集上的损失和准确率
        avg_loss = total_loss / len(test_loader.dataset)
        accuracy = correct / len(test_loader.dataset)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")

torch.save(model.state_dict(), '1DCNN.pt')