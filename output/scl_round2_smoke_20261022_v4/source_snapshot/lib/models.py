import torch
import torch.nn as nn
import numpy as np

class Model(nn.Module):
    def __init__(self, input_shape, n_actions, conv_units=128, dense_units=512):  # 增大了卷积核还有神经元的数目，缓解加入Dropout后训练前期欠拟合
        super(Model, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], conv_units, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(conv_units, conv_units, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(conv_units, conv_units, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, dense_units),
            nn.ReLU(),
            nn.Dropout(0.3),  # 运用了Dropout
            nn.Linear(dense_units, dense_units),
            nn.ReLU(),
            nn.Dropout(0.3),  # 运用了Dropout
            nn.Linear(dense_units, n_actions)
        )
        # self.loss = nn.MSELoss()
        self.loss = nn.BCEWithLogitsLoss()  # 采用BCE损失函数
        # self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([6]))

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

    def compute_loss(self, predictions, targets, mask=None):
        # predictions: raw logits
        # targets: 0/1 float targets
        # mask: boolean mask for unknown positions (optional)
        targets = targets.to(predictions.dtype)

        if mask is not None:
            predictions = predictions[mask]
            targets = targets[mask]

        if isinstance(self.loss, nn.BCEWithLogitsLoss):
            return self.loss(predictions, targets)
        elif isinstance(self.loss, nn.MSELoss):
            return self.loss(predictions, targets)
        else:
            return self.loss(predictions, targets)
