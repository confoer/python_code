import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
import math
from typing import List, Dict, Tuple, Optional


# 锚框生成器
class AnchorGenerator(nn.Module):
    def __init__(
        self,
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    ):
        super(AnchorGenerator, self).__init__()
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        
    def generate_anchors(self, feature_map_size, stride):
        h, w = feature_map_size
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 生成中心坐标
        y = torch.arange(0, h, dtype=torch.float32, device=device)
        x = torch.arange(0, w, dtype=torch.float32, device=device)
        y, x = torch.meshgrid(y, x)
        
        y = y.flatten()
        x = x.flatten()
        
        # 生成锚框
        anchors = []
        for size, aspect_ratio in zip(self.sizes, self.aspect_ratios):
            for s in size:
                for ar in aspect_ratio:
                    h_anchor = s * math.sqrt(ar)
                    w_anchor = s / math.sqrt(ar)
                    
                    # 计算锚框坐标 (xmin, ymin, xmax, ymax)
                    boxes = torch.stack(
                        [
                            x - w_anchor / 2,
                            y - h_anchor / 2,
                            x + w_anchor / 2,
                            y + h_anchor / 2
                        ],
                        dim=1
                    )
                    anchors.append(boxes)
        
        return torch.cat(anchors, dim=1)
    
    def forward(self, feature_maps):
        anchors = []
        for i, feature_map in enumerate(feature_maps):
            height, width = feature_map.shape[-2], feature_map.shape[-1]
            stride = 2 ** (i + 3)  # 假设特征图对应stride为8, 16, 32, 64, 128
            anchors.append(self.generate_anchors((height, width), stride))
        
        return anchors


# RetinaNet的分类子网络
class ClassificationSubnet(nn.Module):
    def __init__(self, num_classes, num_anchors=9):
        super(ClassificationSubnet, self).__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # 4个3x3卷积层，每个后接ReLU
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        
        # 最终的卷积层，输出分类结果
        self.conv_cls = nn.Conv2d(256, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        cls = self.conv_cls(x)
        return cls


# RetinaNet的回归子网络
class RegressionSubnet(nn.Module):
    def __init__(self, num_anchors=9):
        super(RegressionSubnet, self).__init__()
        
        self.num_anchors = num_anchors
        
        # 4个3x3卷积层，每个后接ReLU
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        
        # 最终的卷积层，输出回归结果 (4个坐标偏移量)
        self.conv_reg = nn.Conv2d(256, num_anchors * 4, kernel_size=3, stride=1, padding=1)
        
        # 初始化权重
        self.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        reg = self.conv_reg(x)
        return reg


# Focal Loss实现
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, num_classes=80):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
    
    def forward(self, inputs, targets):
        # inputs: [batch, num_anchors * num_classes, h, w]
        # targets: [batch, num_anchors, 1] 其中-1表示忽略，0表示背景，>0表示类别索引
        
        batch_size = inputs.shape[0]
        num_anchors = inputs.shape[1] // self.num_classes
        inputs = inputs.view(batch_size, num_anchors, self.num_classes)
        targets = targets.view(batch_size, num_anchors)
        
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(inputs.view(-1, self.num_classes), targets.view(-1), reduction='none')
        ce_loss = ce_loss.view(batch_size, num_anchors)
        
        # 计算pt (probability for target class)
        pt = torch.exp(-ce_loss)
        
        # 计算focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # 只计算正样本和负样本的损失，忽略标记为-1的锚框
        pos_mask = targets > 0
        neg_mask = targets == 0
        focal_loss = focal_loss * (pos_mask.float() + neg_mask.float())
        
        # 计算平均损失
        loss = focal_loss.sum() / (pos_mask.float().sum() + neg_mask.float().sum() + 1e-6)
        return loss


# Smooth L1 Loss for bounding box regression
class SmoothL1Loss(nn.Module):
    def __init__(self):
        super(SmoothL1Loss, self).__init__()
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
    
    def forward(self, inputs, targets, anchors, pos_mask):
        # inputs: [batch, num_anchors * 4, h, w]
        # targets: [batch, num_anchors, 4]
        # anchors: [num_anchors, 4]
        # pos_mask: [batch, num_anchors]
        
        batch_size = inputs.shape[0]
        num_anchors = inputs.shape[1] // 4
        inputs = inputs.view(batch_size, num_anchors, 4)
        
        # 只计算正样本的损失
        loss = self.smooth_l1(inputs, targets)
        loss = loss * pos_mask.unsqueeze(2).float()
        
        # 计算平均损失
        loss = loss.sum() / (pos_mask.float().sum() * 4 + 1e-6)
        return loss


# RetinaNet主网络
class RetinaNet(nn.Module):
    def __init__(self, num_classes=80):
        super(RetinaNet, self).__init__()
        
        # 加载预训练的ResNet50作为骨干网络
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # 构建FPN
        self.fpn = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=1),  # C2 -> P2
            nn.Conv2d(512, 256, kernel_size=1),  # C3 -> P3
            nn.Conv2d(1024, 256, kernel_size=1), # C4 -> P4
            nn.Conv2d(2048, 256, kernel_size=1), # C5 -> P5
            nn.Conv2d(256, 256, kernel_size=3, padding=1)  # P5 -> P6 (下采样)
        ])
        
        # 横向连接的上采样层
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=1),  # P2 lateral
            nn.Conv2d(256, 256, kernel_size=1),  # P3 lateral
            nn.Conv2d(256, 256, kernel_size=1),  # P4 lateral
            nn.Conv2d(256, 256, kernel_size=1)   # P5 lateral
        ])
        
        # 锚框生成器
        self.anchor_generator = AnchorGenerator()
        
        # 分类子网络
        self.cls_subnet = ClassificationSubnet(num_classes)
        
        # 回归子网络
        self.reg_subnet = RegressionSubnet()
        
        # 初始化分类子网络的偏置
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.cls_subnet.conv_cls.bias.data.fill_(bias_value)
    
    def forward(self, x):
        # 提取ResNet特征
        features = self.backbone.conv1(x)
        features = self.backbone.bn1(features)
        features = self.backbone.relu(features)
        features = self.backbone.maxpool(features)
        
        c2 = self.backbone.layer1(features)
        c3 = self.backbone.layer2(c2)
        c4 = self.backbone.layer3(c3)
        c5 = self.backbone.layer4(c4)
        
        # 构建FPN
        p5 = self.fpn[3](c5)
        p4 = self.fpn[2](c4) + F.interpolate(p5, size=p4.shape[-2:], mode='nearest')
        p3 = self.fpn[1](c3) + F.interpolate(p4, size=p3.shape[-2:], mode='nearest')
        p2 = self.fpn[0](c2) + F.interpolate(p3, size=p2.shape[-2:], mode='nearest')
        p6 = self.fpn[4](p5)  # 下采样得到P6
        
        # 横向连接优化
        p2 = self.lateral_convs[0](p2)
        p3 = self.lateral_convs[1](p3)
        p4 = self.lateral_convs[2](p4)
        p5 = self.lateral_convs[3](p5)
        
        # 最终的特征图
        feature_maps = [p2, p3, p4, p5, p6]
        
        # 生成锚框
        anchors = self.anchor_generator(feature_maps)
        
        # 分类和回归预测
        cls_outputs = []
        reg_outputs = []
        for feature_map in feature_maps:
            cls_outputs.append(self.cls_subnet(feature_map))
            reg_outputs.append(self.reg_subnet(feature_map))
        
        return {
            'cls_outputs': cls_outputs,
            'reg_outputs': reg_outputs,
            'anchors': anchors
        }


# 训练时的目标匹配和损失计算
def compute_losses(
    outputs: Dict[str, List[torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    num_classes: int = 80
) -> Dict[str, torch.Tensor]:
    cls_outputs = outputs['cls_outputs']
    reg_outputs = outputs['reg_outputs']
    anchors = outputs['anchors']
    
    focal_loss = FocalLoss(num_classes=num_classes)
    smooth_l1_loss = SmoothL1Loss()
    
    cls_losses = []
    reg_losses = []
    
    for i, target in enumerate(targets):
        boxes = target['boxes']
        labels = target['labels']
        
        # 锚框与真实框的匹配
        # 这里需要实现锚框匹配逻辑，包括计算IoU，分配正负样本等
        # 为简化示例，这里省略具体实现，实际应用中需要完整实现
        
        # 假设我们已经有了每个锚框的目标类别和回归目标
        # targets_cls: [num_anchors]，-1表示忽略，0表示背景，>0表示类别索引
        # targets_reg: [num_anchors, 4]
        
        # 为了示例，这里使用随机目标
        batch_size = cls_outputs[0].shape[0]
        num_anchors = anchors[0].shape[1]
        targets_cls = torch.randint(0, num_classes + 1, (batch_size, num_anchors), device=cls_outputs[0].device)
        targets_reg = torch.randn(batch_size, num_anchors, 4, device=reg_outputs[0].device)
        pos_mask = targets_cls > 0
        
        # 计算分类损失
        cls_inputs = torch.cat([o.flatten(2) for o in cls_outputs], dim=2)
        cls_targets = targets_cls
        cls_loss = focal_loss(cls_inputs, cls_targets)
        cls_losses.append(cls_loss)
        
        # 计算回归损失
        reg_inputs = torch.cat([o.flatten(2) for o in reg_outputs], dim=2)
        reg_targets = targets_reg
        reg_loss = smooth_l1_loss(reg_inputs, reg_targets, anchors[0], pos_mask)
        reg_losses.append(reg_loss)
    
    # 计算平均损失
    loss_dict = {
        'cls_loss': torch.stack(cls_losses).mean(),
        'reg_loss': torch.stack(reg_losses).mean()
    }
    loss_dict['total_loss'] = loss_dict['cls_loss'] + loss_dict['reg_loss']
    
    return loss_dict


# 推理时的后处理函数
def inference(
    outputs: Dict[str, List[torch.Tensor]],
    image_size: Tuple[int, int],
    score_threshold: float = 0.05,
    nms_threshold: float = 0.5,
    max_detections: int = 100
) -> List[Dict[str, torch.Tensor]]:
    cls_outputs = outputs['cls_outputs']
    reg_outputs = outputs['reg_outputs']
    anchors = outputs['anchors']
    
    results = []
    
    for i in range(cls_outputs[0].shape[0]):  # 遍历批次中的每个图像
        cls_scores = []
        reg_deltas = []
        
        # 收集所有特征图的预测结果
        for cls_out, reg_out in zip(cls_outputs, reg_outputs):
            cls_scores.append(cls_out[i].permute(1, 2, 0).reshape(-1, num_classes))
            reg_deltas.append(reg_out[i].permute(1, 2, 0).reshape(-1, 4))
        
        cls_scores = torch.cat(cls_scores, dim=0)
        reg_deltas = torch.cat(reg_deltas, dim=0)
        all_anchors = torch.cat(anchors, dim=1)[i]
        
        # 应用得分阈值
        max_scores, max_classes = torch.max(cls_scores, dim=1)
        keep = max_scores > score_threshold
        
        scores = max_scores[keep]
        classes = max_classes[keep]
        deltas = reg_deltas[keep]
        anchors_filtered = all_anchors[keep]
        
        # 解码回归预测，得到边界框
        boxes = decode_boxes(anchors_filtered, deltas)
        
        # 非极大值抑制(NMS)
        keep_indices = nms(boxes, scores, nms_threshold)
        keep_indices = keep_indices[:max_detections]
        
        results.append({
            'boxes': boxes[keep_indices],
            'labels': classes[keep_indices],
            'scores': scores[keep_indices]
        })
    
    return results


# 边界框解码函数
def decode_boxes(anchors: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    # 锚框坐标 (xmin, ymin, xmax, ymax)
    anchor_widths = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]
    anchor_ctr_x = anchors[:, 0] + 0.5 * anchor_widths
    anchor_ctr_y = anchors[:, 1] + 0.5 * anchor_heights
    
    # 预测的偏移量
    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]
    
    # 解码公式
    pred_ctr_x = dx * anchor_widths + anchor_ctr_x
    pred_ctr_y = dy * anchor_heights + anchor_ctr_y
    pred_w = torch.exp(dw) * anchor_widths
    pred_h = torch.exp(dh) * anchor_heights
    
    # 转换为(xmin, ymin, xmax, ymax)格式
    pred_boxes = torch.zeros_like(anchors)
    pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h
    
    return pred_boxes


# 非极大值抑制
def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    _, order = scores.sort(0, descending=True)
    
    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()
            keep.append(i)
        
        # 计算IoU
        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        
        w = (xx2 - xx1 + 1).clamp(min=0)
        h = (yy2 - yy1 + 1).clamp(min=0)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        # 保留IoU小于阈值的框
        idx = (iou <= iou_threshold).nonzero(as_tuple=False).squeeze()
        if idx.numel() == 0:
            break
        order = order[idx + 1]
    
    return torch.tensor(keep, device=boxes.device)