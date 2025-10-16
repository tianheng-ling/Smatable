import torch.nn as nn


class Float2DCNN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        # in_channels, le_classes, input_height, input_width

        in_channels = kwargs.get("in_channels")
        out_channels = 64  # kwargs.get("out_channels")
        input_height = kwargs.get("input_height")
        input_width = kwargs.get("input_width")
        le_classes = kwargs.get("le_classes")
        p = kwargs.get("p")

        # Block 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2 (with residual)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3
        self.conv5 = nn.Conv2d(out_channels, 8 * out_channels, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        h = input_height // 8
        w = input_width // 8
        flatten_dim = 8 * out_channels * h * w
        self.fc1 = nn.Linear(in_features=flatten_dim, out_features=4 * out_channels)
        self.dropout = nn.Dropout(p=p)
        self.fc2 = nn.Linear(in_features=4 * out_channels, out_features=le_classes)

    def forward(self, x):
        x = x.unsqueeze(1)

        # Block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool1(x)

        residual = x

        # Block 2
        x = self.conv3(x)
        x = self.conv4(x)
        x = x + residual
        x = self.relu3(x)
        x = self.pool2(x)

        # Block 3
        x = self.conv5(x)
        x = self.relu4(x)
        x = self.pool3(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
