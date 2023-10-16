import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)  # Ajout de kernels (32) et réduction de la taille du noyau (3x3)
        self.conv2 = nn.Conv2d(32, 64, 5)  # Ajout de kernels (64) et réduction de la taille du noyau (3x3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)  # Ajustement de l'entrée de la couche fully connected
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x