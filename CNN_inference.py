import os
import csv
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

import numpy as np

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN1D(8)

model = model.to(device)

model.load_state_dict(torch.load("tut2-model.pt"))

test_data = []
test_labels = []
batch_size = 256

''' 1 test '''
'''
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
                test.append(row_data)
                test_labels.append(1)
'''
# 3 test
'''
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
# 2 test
'''
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


model.eval()

test_data = np.array(test_data)
test_labels = np.array(test_labels)

test_data = torch.Tensor(test_data).unsqueeze(1)
test_labels = torch.LongTensor(test_labels)

test_dataset = TensorDataset(test_data, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        ver_outputs = model(inputs)

print(ver_outputs.argmax(1))