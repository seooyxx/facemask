import torch
from torch import nn
from torchvision import models
models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

class MasksModelFromResNet(nn.Module):
    def __init__(self, num_classes, pretrained=True, train_all_layers=True):
        super().__init__()

        weights = None
        if pretrained:
           weights = models.resnet34().state_dict()

        self.base_model = models.resnet34(weights=weights)

        if not train_all_layers:
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.base_model.fc = nn.Linear(in_features=512, out_features=1)

    def forward(self, x):
        return nn.Sigmoid()(self.base_model.forward(x))

    def to(self, device):
        new_model = super().to(device)
        new_model.base_model = new_model.base_model.to(device)
        return new_model
