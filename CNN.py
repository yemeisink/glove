import os
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

batch_size = 32
learning_rate = 0.001
num_epochs = 25
train_data = []
train_labels = []

# read 1 disk
train_new_path = os.path.join(os.getcwd(), "train_new")
one_path = os.path.join(train_new_path, "1")
csv_files = [os.path.join(one_path, "2023_0727_15_06_21.csv"),
             os.path.join(one_path, "2023_0731_17_36_39.csv"),
             os.path.join(one_path, "2023_0727_15_16_48.csv"),
             os.path.join(one_path, "2023_0731_19_09_46.csv")]

for csv_file in csv_files:
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) > 1:
                row_data = list(map(int, row[1:-1]))
                train_data.append(row_data)
                train_labels.append(1)

# read 2 disk
train_new_path = os.path.join(os.getcwd(), "train_new")
one_path = os.path.join(train_new_path, "2")
csv_files = [os.path.join(one_path, "2023_0727_15_07_54.csv"),
             os.path.join(one_path, "2023_0727_15_17_20.csv"),
             os.path.join(one_path, "2023_0731_17_37_51.csv"),
             os.path.join(one_path, "2023_0731_19_10_54.csv")
             ]

for csv_file in csv_files:
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) > 1:
                row_data = list(map(int, row[1:-1]))
                train_data.append(row_data)
                train_labels.append(2)

# read 3 disk
train_new_path = os.path.join(os.getcwd(), "train_new")
one_path = os.path.join(train_new_path, "3")
csv_files = [os.path.join(one_path, "2023_0727_15_09_00.csv"),
             os.path.join(one_path, "2023_0731_17_41_01.csv"),
             os.path.join(one_path, "2023_0727_15_18_03.csv"),
             os.path.join(one_path, "2023_0731_19_11_50.csv")
             ]

for csv_file in csv_files:
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) > 1:
                row_data = list(map(int, row[1:-1]))
                train_data.append(row_data)
                train_labels.append(3)

# read 4 disk
train_new_path = os.path.join(os.getcwd(), "train_new")
one_path = os.path.join(train_new_path, "4")
csv_files = [os.path.join(one_path, "2023_0727_15_10_19.csv"),
             os.path.join(one_path, "2023_0727_15_18_53.csv"),
             os.path.join(one_path, "2023_0731_18_31_02.csv"),
             os.path.join(one_path, "2023_0731_19_13_25.csv")
             ]

for csv_file in csv_files:
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) > 50:
                row_data = list(map(int, row[1:-1]))
                train_data.append(row_data)
                train_labels.append(4)

# read 5 disk
train_new_path = os.path.join(os.getcwd(), "train_new")
one_path = os.path.join(train_new_path, "5")
csv_files = [os.path.join(one_path, "2023_0727_15_12_25.csv"),
             os.path.join(one_path, "2023_0731_18_37_06.csv"),
             os.path.join(one_path, "2023_0727_15_20_08.csv"),
             os.path.join(one_path, "2023_0731_19_15_43.csv")
             ]

for csv_file in csv_files:
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) > 50:
                row_data = list(map(int, row[1:-1]))
                train_data.append(row_data)
                train_labels.append(5)

# read 6 disk
train_new_path = os.path.join(os.getcwd(), "train_new")
one_path = os.path.join(train_new_path, "6")
csv_files = [os.path.join(one_path, "2023_0727_15_13_31.csv"),
             os.path.join(one_path, "2023_0727_15_20_52.csv"),
             os.path.join(one_path, "2023_0731_18_39_20.csv"),
             os.path.join(one_path, "2023_0731_19_17_10.csv")
             ]

for csv_file in csv_files:
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) > 50:
                row_data = list(map(int, row[1:-1]))
                train_data.append(row_data)
                train_labels.append(6)

# read 7 disk
train_new_path = os.path.join(os.getcwd(), "train_new")
one_path = os.path.join(train_new_path, "7")
csv_files = [os.path.join(one_path, "2023_0727_15_14_53.csv"),
             os.path.join(one_path, "2023_0731_18_41_12.csv"),
             os.path.join(one_path, "2023_0727_15_21_31.csv"),
             os.path.join(one_path, "2023_0731_19_21_53.csv")
             ]

for csv_file in csv_files:
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) > 1:
                row_data = list(map(int, row[1:-1]))
                train_data.append(row_data)
                train_labels.append(7)

# read bare condition
train_new_path = os.path.join(os.getcwd(), "train_new")
one_path = os.path.join(train_new_path, "bare")
csv_files = [os.path.join(one_path, "2023_0727_15_22_55.csv"),
             os.path.join(one_path, "2023_0731_18_44_52.csv")]

for csv_file in csv_files:
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) > 1:
                row_data = list(map(int, row[1:-1]))
                train_data.append(row_data)
                train_labels.append(0)

''' Test data '''

test_data = []
test_labels = []


# 1 test
test_new_path = os.path.join(os.getcwd(), "test_new")
one_path = os.path.join(test_new_path, "1")
csv_files = [os.path.join(one_path, "2023_0727_15_16_48.csv"),
             os.path.join(one_path, "2023_0731_19_09_46.csv")]

for csv_file in csv_files:
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) > 1:
                row_data = list(map(int, row[1:-1]))
                test_data.append(row_data)
                test_labels.append(1)

# 3 test
test_new_path = os.path.join(os.getcwd(), "test_new")
one_path = os.path.join(test_new_path, "3")
csv_files = [os.path.join(one_path, "2023_0727_15_18_03.csv"),
             os.path.join(one_path, "2023_0731_19_11_50.csv")]

for csv_file in csv_files:
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) > 1:
                row_data = list(map(int, row[1:-1]))
                test_data.append(row_data)
                test_labels.append(3)

# 5 test
test_new_path = os.path.join(os.getcwd(), "test_new")
one_path = os.path.join(test_new_path, "5")
csv_files = [os.path.join(one_path, "2023_0727_15_20_08.csv"),
             os.path.join(one_path, "2023_0731_19_15_43.csv")]

for csv_file in csv_files:
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) > 1:
                row_data = list(map(int, row[1:-1]))
                test_data.append(row_data)
                test_labels.append(5)

# 7 test
test_new_path = os.path.join(os.getcwd(), "test_new")
one_path = os.path.join(test_new_path, "7")
csv_files = [os.path.join(one_path, "2023_0727_15_21_31.csv"),
             os.path.join(one_path, "2023_0731_19_21_53.csv")]

for csv_file in csv_files:
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) > 1:
                row_data = list(map(int, row[1:-1]))
                test_data.append(row_data)
                test_labels.append(7)

# bare test
test_new_path = os.path.join(os.getcwd(), "test_new")
one_path = os.path.join(test_new_path, "bare")
csv_files = [os.path.join(one_path, "2023_0727_15_22_23.csv"),
             os.path.join(one_path, "2023_0731_19_08_52.csv")]

for csv_file in csv_files:
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) > 1:
                row_data = list(map(int, row[1:-1]))
                test_data.append(row_data)
                test_labels.append(0)



# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_index, test_index in kf.split(train_data):
    train_data_fold = [train_data[i] for i in train_index]
    train_labels_fold = [train_labels[i] for i in train_index]

    test_data_fold = [train_data[i] for i in test_index]
    test_labels_fold = [train_labels[i] for i in test_index]

    # put data to numpy

    train_data_fold = np.array(train_data_fold)
    train_labels_fold = np.array(train_labels_fold)

    test_data_fold = np.array(test_data_fold)
    test_labels_fold = np.array(test_labels_fold)

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
            nn.Linear(480, 240),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(240, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)

        return x
'''
class CNN1D(nn.Module):
    def __init__(self, num_classes):

        super().__init__()
        self.layers_1 = nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Linear(14, 256),
        nn.ReLU()
        )
        self.layers_2 = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(256 * 128, num_classes)
        )

    def forward(self, x):
        x = self.layers_1(x)
        x = x.view(x.size(0), -1)
        x = self.layers_2(x)

        return x
'''

# Assuming input size (sequence length) is 70 and number of classes is 8

num_classes = 8

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

torch.save(model.state_dict(), 'tut3-model.pt')