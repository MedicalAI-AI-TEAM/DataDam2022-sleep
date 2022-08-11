# sys.path.append('/workspace/KD_InductiveBias/core')
import torch.nn as nn
import torch


class LinearProjection(nn.Module):
    def __init__(self, d_input=512, d_output=1):
        super(LinearProjection, self).__init__()
        self.linear = nn.Linear(d_input, d_output)

    def forward(self, x):
        output = self.linear(x)
        return output


class ResNetAndrewNg(nn.Module):
    def __init__(self, config):
        super(ResNetAndrewNg, self).__init__()
        self.encoder = nn.Sequential(
            ResNetBlockType1(300, 32, config["kernel_size"], config["dropout"]),
            ResNetBlockType2(32, 32, config["kernel_size"], 1, config["dropout"]),
            ResNetBlockType2(32, 32, config["kernel_size"], 2, config["dropout"]),
            ResNetBlockType2(32, 64, config["kernel_size"], 1, config["dropout"]),
            ResNetBlockType2(64, 64, config["kernel_size"], 2, config["dropout"]),
            ResNetBlockType2(64, 64, config["kernel_size"], 1, config["dropout"]),
            ResNetBlockType2(64, 64, config["kernel_size"], 2, config["dropout"]),
            ResNetBlockType2(64, 128, config["kernel_size"], 1, config["dropout"]),
            ResNetBlockType2(128, 128, config["kernel_size"], 2, config["dropout"]),
            ResNetBlockType2(128, 128, config["kernel_size"], 1, config["dropout"]),
            ResNetBlockType2(128, 128, config["kernel_size"], 2, config["dropout"]),
            ResNetBlockType2(128, 256, config["kernel_size"], 1, config["dropout"]),
            ResNetBlockType2(256, 256, config["kernel_size"], 2, config["dropout"]),
            ResNetBlockType2(256, 256, config["kernel_size"], 1, config["dropout"]),
            ResNetBlockType2(256, 256, config["kernel_size"], 2, config["dropout"]),
            ResNetBlockType2(256, 512, config["kernel_size"], 1, config["dropout"]),
        )

        self.fc = LinearProjection(512, config["output_size"])

    def forward(self, x):
        encoded = self.encoder(x)
        encoded = encoded.mean(axis=2)
        output = self.fc(encoded)
        output = nn.functional.softmax(output, dim=1)
        return output


class ResNetBlockType1(nn.Module):
    def __init__(self, in_channels, out_channels=64, kernel_size=15, dropout=0.1):
        super(ResNetBlockType1, self).__init__()
        # kernel_size = 15
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            padding_mode="zeros",
            dilation=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(
            num_features=out_channels, eps=1e-05, momentum=0.1, affine=True
        )
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            padding_mode="zeros",
            dilation=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(
            num_features=out_channels, eps=1e-05, momentum=0.1, affine=True
        )
        self.conv3 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=(kernel_size - 1) // 2,
            padding_mode="zeros",
            dilation=1,
            bias=False,
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv1_bn1 = self.bn1(conv1)
        conv1_bn1_relu = self.relu(conv1_bn1)
        max_pool = self.pool(conv1_bn1_relu)

        conv2 = self.conv2(conv1_bn1_relu)
        conv2_bn2 = self.bn2(conv2)
        conv2_bn2_relu = self.relu(conv2_bn2)
        conv2_bn2_relu_dropout = self.dropout(conv2_bn2_relu)
        conv3 = self.conv3(conv2_bn2_relu_dropout)

        outputs = conv3 + max_pool
        return outputs


class ResNetBlockType2(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, subsample_factor, dropout
    ):
        super(ResNetBlockType2, self).__init__()
        # kernel_size = 15

        self.bn1 = nn.BatchNorm1d(
            num_features=in_channels, eps=1e-05, momentum=0.1, affine=True
        )
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            padding_mode="zeros",
            dilation=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(
            num_features=out_channels, eps=1e-05, momentum=0.1, affine=True
        )
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=subsample_factor,
            padding=(kernel_size - 1) // 2,
            padding_mode="zeros",
            dilation=1,
            bias=False,
        )
        self.conv_eye = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            padding_mode="zeros",
            dilation=1,
            bias=False,
        )
        self.pool = nn.MaxPool1d(
            kernel_size=subsample_factor,
            stride=subsample_factor,
            padding=0,
            ceil_mode=True,
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs):
        max_pool = self.pool(inputs)
        max_pool_conv = self.conv_eye(max_pool)

        bn1 = self.bn1(inputs)
        bn1_relu = self.relu(bn1)
        bn1_relu_conv1 = self.conv1(bn1_relu)

        bn2 = self.bn2(bn1_relu_conv1)
        bn2_relu = self.relu(bn2)
        bn2_relu_drop = self.dropout(bn2_relu)
        bn2_relu_drop_conv2 = self.conv2(bn2_relu_drop)
        outputs = max_pool_conv + bn2_relu_drop_conv2

        return outputs
