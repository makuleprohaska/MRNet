import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import AlexNet_Weights

class MRNet3(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.model1 = models.alexnet(weights=AlexNet_Weights.DEFAULT)
        self.model2 = models.alexnet(weights=AlexNet_Weights.DEFAULT)
        self.model3 = models.alexnet(weights=AlexNet_Weights.DEFAULT)
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Add dropout for each view's features
        self.dropout_view1 = nn.Dropout(p=0.3)
        self.dropout_view2 = nn.Dropout(p=0.3)
        self.dropout_view3 = nn.Dropout(p=0.3)

        self.classifier1 = nn.Linear(int(256*3), 256)
        self.dropout = nn.Dropout(p=0.5)
        self.activation = nn.ReLU() 
        self.classifier2 = nn.Linear(256, 1)

        #self.classifier =  nn.Linear(int(256*3), 1)

    def forward(self, x):
        
        x_1 = x[0]
        x_2 = x[1]
        x_3 = x[2]
        
        x_1 = torch.squeeze(x_1, dim=0) # only batch size 1 supported
        x_1 = self.model1.features(x_1)
        x_1 = self.gap(x_1).view(x_1.size(0), -1)
        x_1 = torch.max(x_1, 0, keepdim=True)[0]    #grok said this is redundant, check
        x_1 = self.dropout_view1(x_1)  # Apply dropout to view features

        x_2 = torch.squeeze(x_2, dim=0) # only batch size 1 supported
        x_2 = self.model2.features(x_2)
        x_2 = self.gap(x_2).view(x_2.size(0), -1)
        x_2 = torch.max(x_2, 0, keepdim=True)[0]
        x_2 = self.dropout_view2(x_2)

        x_3 = torch.squeeze(x_3, dim=0) # only batch size 1 supported
        x_3 = self.model3.features(x_3)
        x_3 = self.gap(x_3).view(x_3.size(0), -1)
        x_3 = torch.max(x_3, 0, keepdim=True)[0]
        x_3 = self.dropout_view3(x_3)
        
        x_stacked = torch.cat((x_1, x_2, x_3), dim=1)
        
        x_stacked = self.classifier1(x_stacked)
        x_stacked = self.dropout(x_stacked)
        x_stacked = self.activation(x_stacked)
        x_stacked = self.classifier2(x_stacked)
        #x_stacked = self.classifier(x_stacked)
        
        return x_stacked 
