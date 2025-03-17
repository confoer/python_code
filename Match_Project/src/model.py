import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class EmotionNet(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.base_model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
            
        # 修改分类头
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes))
        
    def forward(self, x):
        return self.base_model(x)