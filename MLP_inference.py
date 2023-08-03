import os
import csv
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

import numpy as np

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP(70, 8)

model = model.to(device)

model.load_state_dict(torch.load("tut1-model.pt"))

ver_data = []
ver_labels = []
batch_size = 256

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

model.eval()

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