import torch.nn as nn
import torch

class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(1024, 1024)
        self.bm1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 1024)
        self.bm2 = nn.BatchNorm1d(1024)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bm1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bm2(out)
        out = self.relu(out)
        out = self.dropout(out)
        return x + out


class Model(nn.Module):
    def __init__(self, input_size, output_size, num_blocks):
        super(Model, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.num_blocks = num_blocks

        self.fc1 = nn.Linear(self.input_size, 1024)
        self.bm1 = nn.BatchNorm1d(1024)

        self.blocks = []
        for block in range(num_blocks):
            self.blocks.append(ResidualBlock())
        self.blocks = nn.ModuleList(self.blocks)

        self.fc2 = nn.Linear(1024, self.output_size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bm1(out)
        out = self.relu(out)
        out = self.dropout(out)

        for i in range(self.num_blocks):
            out = self.res_blocks[i](out)
        out = self.fc2(out)
        return out








