import os
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

batch_size = 256
learning_rate = 0.001
num_epochs = 32
train_data = []
train_labels = []

# read 1 disk
train_new_path = os.path.join(os.getcwd(), "train_new")
one_path = os.path.join(train_new_path, "1")
csv_files = [os.path.join(one_path, "2023_0727_15_06_21.csv"),
             os.path.join(one_path, "2023_0731_17_36_39.csv")]

for csv_file in csv_files:
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) > 1:
                row_data = list(map(int, row[1:-1]))
                train_data.append(row_data)
                train_labels.append(1)

'''
# read 2 disk
train_new_path = os.path.join(os.getcwd(), "train_new")
one_path = os.path.join(train_new_path, "2")
csv_files = [os.path.join(one_path, "2023_0727_15_07_54.csv"),
             os.path.join(one_path, "2023_0731_17_37_51.csv")]

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
             os.path.join(one_path, "2023_0731_17_41_01.csv")]

for csv_file in csv_files:
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) > 1:
                row_data = list(map(int, row[1:-1]))
                train_data.append(row_data)
                train_labels.append(3)
'''

# read 4 disk
train_new_path = os.path.join(os.getcwd(), "train_new")
one_path = os.path.join(train_new_path, "4")
csv_files = [os.path.join(one_path, "2023_0727_15_10_19.csv"),
             os.path.join(one_path, "2023_0731_18_31_02.csv")]

for csv_file in csv_files:
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) > 1:
                row_data = list(map(int, row[1:-1]))
                train_data.append(row_data)
                train_labels.append(4)

'''
# read 5 disk
train_new_path = os.path.join(os.getcwd(), "train_new")
one_path = os.path.join(train_new_path, "5")
csv_files = [os.path.join(one_path, "2023_0727_15_12_25.csv"),
             os.path.join(one_path, "2023_0731_18_37_06.csv")]

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
             os.path.join(one_path, "2023_0731_18_39_20.csv")]

for csv_file in csv_files:
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) > 1:
                row_data = list(map(int, row[1:-1]))
                train_data.append(row_data)
                train_labels.append(6)
'''

# read 7 disk
train_new_path = os.path.join(os.getcwd(), "train_new")
one_path = os.path.join(train_new_path, "7")
csv_files = [os.path.join(one_path, "2023_0727_15_14_53.csv"),
             os.path.join(one_path, "2023_0731_18_41_12.csv")]

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

'''
# 2 test
test_new_path = os.path.join(os.getcwd(), "test_new")
one_path = os.path.join(test_new_path, "2")
csv_files = [os.path.join(one_path, "2023_0727_15_17_20.csv"),
             os.path.join(one_path, "2023_0731_19_10_54.csv")]

for csv_file in csv_files:
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) > 1:
                row_data = list(map(int, row[1:-1]))
                test_data.append(row_data)
                test_labels.append(2)


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
'''
# 4 test
test_new_path = os.path.join(os.getcwd(), "test_new")
one_path = os.path.join(test_new_path, "4")
csv_files = [os.path.join(one_path, "2023_0727_15_18_53.csv"),
             os.path.join(one_path, "2023_0731_19_13_25.csv")]

for csv_file in csv_files:
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) > 1:
                row_data = list(map(int, row[1:-1]))
                test_data.append(row_data)
                test_labels.append(4)

'''
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


# 6 test
test_new_path = os.path.join(os.getcwd(), "test_new")
one_path = os.path.join(test_new_path, "6")
csv_files = [os.path.join(one_path, "2023_0727_15_20_52.csv"),
             os.path.join(one_path, "2023_0731_19_17_10.csv")]

for csv_file in csv_files:
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) > 1:
                row_data = list(map(int, row[1:-1]))
                test_data.append(row_data)
                test_labels.append(6)
'''

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

# put data to numpy
train_data = np.array(train_data)
train_labels = np.array(train_labels)

test_data = np.array(test_data)
test_labels = np.array(test_labels)

# tensor
train_data = torch.Tensor(train_data)
train_labels = torch.LongTensor(train_labels)

test_data = torch.Tensor(test_data)
test_labels = torch.LongTensor(test_labels)

'''创建迭代器dataloader'''
train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(test_data, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        x = self.feature(x)

        return x

input_size = 70
output_size = 8

model = MLP(input_size, output_size)

'''构建优化器'''
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

'''损失函数'''
criterion = nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''gpu训练'''
model = model.to(device)
criterion = criterion.to(device)


for epoch in range(num_epochs):
    model.train()  # 设置为训练模式
    total_loss = 0
    correct = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()  # 梯度清零
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()  # 反向传播
        optimizer.step()  # 参数更新
        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == targets).sum().item()

    # 打印训练集上的损失和准确率
    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = correct / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.4f}")

    # 在测试集上评估模型
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
        print(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")


torch.save(model.state_dict(), 'tut2-model.pt')

ver_data = []
ver_labels = []

''' 1 test '''
ver_new_path = os.path.join(os.getcwd(), "test_new")
one_path = os.path.join(ver_new_path, "1")
csv_files = [os.path.join(one_path, "2023_0727_15_16_48.csv"),
             os.path.join(one_path, "2023_0731_19_09_46.csv")]


for csv_file in csv_files:
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) > 1:
                row_data = list(map(int, row[1:-1]))
                ver_data.append(row_data)
                ver_labels.append(1)

''' 4 test '''
'''
ver_new_path = os.path.join(os.getcwd(), "test_new")
one_path = os.path.join(ver_new_path, "4")
csv_files = [os.path.join(one_path, "2023_0727_15_18_53.csv"),
             os.path.join(one_path, "2023_0731_19_13_25.csv")]


for csv_file in csv_files:
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) > 1:
                row_data = list(map(int, row[1:-1]))
                ver_data.append(row_data)
                ver_labels.append(4)
'''
''' 7 test '''
'''
ver_new_path = os.path.join(os.getcwd(), "test_new")
one_path = os.path.join(ver_new_path, "7")
csv_files = [os.path.join(one_path, "2023_0727_15_18_53.csv"),
             os.path.join(one_path, "2023_0731_19_13_25.csv")]


for csv_file in csv_files:
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) > 1:
                row_data = list(map(int, row[1:-1]))
                ver_data.append(row_data)
                ver_labels.append(7)
'''
'''
ver_data = np.array(ver_data)
ver_labels = np.array(ver_labels)

ver_data = torch.Tensor(ver_data)
ver_labels = torch.LongTensor(ver_labels)

ver_dataset = TensorDataset(ver_data, ver_labels)
ver_loader = DataLoader(ver_dataset, batch_size=batch_size)

with torch.no_grad():
    for inputs, targets in ver_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        ver_outputs = model(inputs)

print(ver_outputs.argmax(1))
'''