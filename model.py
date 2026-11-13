import torch
import torch.nn as nn
from torchvision import models

class MyModel(nn.Module):
    
    def __init__(self, backbone='resnet18'):
        super(MyModel, self).__init__()
        
        if backbone == 'resnet18':
            self.model = models.resnet18(weights=None)
            
        elif backbone == 'resnet34':
            self.model = models.resnet34(weights=None)
            
        elif backbone == 'resnet50':
            self.model = models.resnet50(weights=None)
            
        
        self.model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        num_features = self.model.fc.in_features
        
        self.model.fc = nn.Linear(num_features, 2)
        
    def forward(self, x):
        output = self.model(x)
        return output
