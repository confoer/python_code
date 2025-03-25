import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class EmotionNet(nn.Module):
    def __init__(self, num_classes=7,width_multiplier=1.0,add_depth = False):
        super().__init__()
        # self.base_model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.base_model = mobilenet_v2(width_mult=width_multiplier, weights=MobileNet_V2_Weights.DEFAULT) 
        # 修改分类头
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        if add_depth:
            self.extra_layer = nn.Sequential(
                nn.Linear(in_features, num_classes),
                nn.ReLU()
            )
        
    def forward(self, x):
        x = self.base_model(x)
        if hasattr(self,'extra_layer'):
            x = self.extra_layer(x)
        return x