import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

import numpy as np

import csv


batch_size = 256
learning_rate = 0.001
num_epochs = 25
train_data = []
train_labels = []

'''bare data'''
with open('train/bare/2023_0714_20_00_20.csv', 'r') as file:
    reader = csv.reader(file)

    for row in reader:
        row_data = list(map(int, row[1:-1]))
        train_data.append(row_data)
        train_labels.append(0)

'''bottom data'''
with open('train/bottom/2023_0714_19_58_33.csv', 'r') as file:
    reader = csv.reader(file)

    for row in reader:
        row_data = list(map(int, row[1:-1]))
        train_data.append(row_data)
        train_labels.append(1)

'''mid data'''
with open('train/mid/2023_0714_19_56_43.csv', 'r') as file:
    reader = csv.reader(file)

    for row in reader:
        row_data = list(map(int, row[1:-1]))
        train_data.append(row_data)
        train_labels.append(2)

'''top data'''
with open('train/top/2023_0714_19_53_40.csv', 'r') as file:
    reader = csv.reader(file)

    for row in reader:
        row_data = list(map(int, row[1:-1]))
        train_data.append(row_data)
        train_labels.append(3)


'''测试集'''
test_data = []
test_labels = []

with open('test/bare/2023_0714_20_11_24.csv', 'r') as file:
    reader = csv.reader(file)

    for row in reader:
        row_data = list(map(int, row[1:-1]))
        test_data.append(row_data)
        test_labels.append(0)

with open('test/bottom/2023_0714_20_15_21.csv', 'r') as file:
    reader = csv.reader(file)

    for row in reader:
        row_data = list(map(int, row[1:-1]))
        test_data.append(row_data)
        test_labels.append(1)

with open('test/mid/2023_0714_20_14_11.csv', 'r') as file:
    reader = csv.reader(file)

    for row in reader:
        row_data = list(map(int, row[1:-1]))
        test_data.append(row_data)
        test_labels.append(2)

with open('test/top/2023_0714_20_13_00.csv', 'r') as file:
    reader = csv.reader(file)

    rows = list(reader)

    # 遍历每一行数据，忽略最后一行
    for row in rows[:-1]:
        row_data = list(map(int, row[1:-1]))
        test_data.append(row_data)
        test_labels.append(3)

# put data to numpy
train_data = np.array(train_data)
train_labels = np.array(train_labels)

test_data = np.array(test_data)
test_labels = np.array(test_labels)


'''查看数据集分别有多少例子'''

print(f'Number of the training : {len(train_data)}')
print(f'Number of the testing : {len(test_data)}')
# Number of the training : 240
# Number of the testing : 129

train_data = torch.Tensor(train_data)
train_labels = torch.LongTensor(train_labels)

test_data = torch.Tensor(test_data)
test_labels = torch.LongTensor(test_labels)


'''创建迭代器dataloader'''
train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(test_data, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


'''创建MLP模型'''
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_size, 250),
            nn.ReLU(),
            nn.Linear(250, 100),
            nn.ReLU(),
            nn.Linear(100, 4)
        )

    def forward(self, x):
        x = self.feature(x)

        return x

input_size = 70
output_size = 4

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


torch.save(model.state_dict(), 'tut1-model.pt')

'''自定义验证'''

ver_data = []
ver_labels = []

'''
# 最顶部
with open('test/top/2023_0714_20_13_00.csv', 'r') as file:
    reader = csv.reader(file)

    rows = list(reader)

    # 遍历每一行数据，忽略最后一行
    for row in rows[:-1]:
        row_data = list(map(int, row[1:-1]))
        ver_data.append(row_data)
        ver_labels.append(3)
'''

'''
# 空手
with open('test/bare/2023_0714_20_11_24.csv', 'r') as file:
    reader = csv.reader(file)

    for row in reader:
        row_data = list(map(int, row[1:-1]))
        ver_data.append(row_data)
        ver_labels.append(0)
'''

'''
# 最底部
with open('test/bottom/2023_0714_20_15_21.csv', 'r') as file:
    reader = csv.reader(file)

    for row in reader:
        row_data = list(map(int, row[1:-1]))
        ver_data.append(row_data)
        ver_labels.append(1)
'''

# 中间
with open('test/mid/2023_0714_20_14_11.csv', 'r') as file:
    reader = csv.reader(file)

    for row in reader:
        row_data = list(map(int, row[1:-1]))
        ver_data.append(row_data)
        ver_labels.append(2)

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