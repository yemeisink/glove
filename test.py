import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

import numpy as np

import csv

from train_model import MLP

test_data = []
test_labels = []

with open('test/top/2023_0714_20_13_00.csv', 'r') as file:
    reader = csv.reader(file)

    rows = list(reader)

    # 遍历每一行数据，忽略最后一行
    for row in rows[:-1]:
        row_data = list(map(int, row[1:-1]))
        test_data.append(row_data)
        test_labels.append(3)

test_data = np.array(test_data)
test_data = torch.Tensor(test_data)
# print(test_data.shape)


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

model_dict = torch.load('tut1-model.pt')
model = MLP(70, 4)
model.load_state_dict(model_dict)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_data = test_data.to(device)
model = model.to(device)

# 在测试集上评估模型
model.eval()  # 设置为评估模式

with torch.no_grad():
    output = model(test_data)


print(output.argmax(1))
print(output)