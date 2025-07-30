import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    A simple CNN designed for the Office Scenario data.
    Input shape: (Batch, 40, 100, 64) -> (B, C, H, W)
    Output shape: (Batch, 5)
    """
    def __init__(self, config):
        super(SimpleCNN, self).__init__()
        
        input_channels = config['model']['input_channels']
        num_classes = config['model']['num_classes']

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # -> (B, 32, 50, 32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # -> (B, 64, 25, 16)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # -> (B, 128, 12, 8)

        # Flatten the output for the fully connected layer
        # The size is 128 * 12 * 8
        self.flattened_size = 128 * 12 * 8
        
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Convolutional layers
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten and fully connected layers
        x = x.view(-1, self.flattened_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) # Output logits
        
        return x


def get_office_model(config):
    """
    Model factory for the Office Scenario.
    """
    model_name = config['model']['name']
    if model_name == 'simple_cnn':
        return SimpleCNN(config)
    else:
        raise ValueError(f"Unknown office model name: {model_name}") 