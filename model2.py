import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import torch_geometric.nn as geom_nn

# 1. Configuration
NUM_KEYPOINTS = 16  # Example: head(1), neck(1), torso(1), arms(4), legs(4), hands(4), feet(2)
NUM_CLASSES = 10    # Example: head, neck, torso, upper-arm, forearm, thigh, calf, hand, foot, other

# Fixed anatomical skeleton connections (example indices)
EDGE_INDEX = torch.tensor([
    [0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
], dtype=torch.long)

# 2. Feature Extraction Backbone
class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        base = resnet18(pretrained=True)
        self.features = nn.Sequential(*list(base.children())[:-2])
        self.keypoint_conv = nn.Conv2d(512, NUM_KEYPOINTS, kernel_size=1)
        
    def forward(self, x):
        features = self.features(x)  # [batch, 512, 8, 8]
        heatmaps = self.keypoint_conv(features)  # [batch, K, 8, 8]
        return features, heatmaps

# 3. Coordinate Decoder with Soft-Argmax
def soft_argmax(heatmaps):
    beta = 100  # Temperature parameter
    B, K, H, W = heatmaps.shape
    heatmaps = F.softmax(heatmaps.view(B, K, -1)*beta, dim=2).view(B, K, H, W)
    
    # Create coordinate grid
    y_coord = torch.linspace(0, 1, H, device=heatmaps.device).reshape(1, 1, H, 1)
    x_coord = torch.linspace(0, 1, W, device=heatmaps.device).reshape(1, 1, 1, W)
    
    expected_y = (heatmaps * y_coord).sum(dim=[2,3])
    expected_x = (heatmaps * x_coord).sum(dim=[2,3])
    return torch.stack([expected_x, expected_y], dim=2)  # [B, K, 2]

# 4. Graph Neural Network (Edge-Augmented GCN)
class EGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = geom_nn.GCNConv(514, 256)  # 512 features + 2 coordinates
        self.conv2 = geom_nn.GCNConv(256, 128)
        self.reg_head = nn.Linear(128, 2)
        self.cls_head = nn.Linear(128, NUM_CLASSES)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.reg_head(x), self.cls_head(x)

# 5. Full Pipeline
class PosePipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.gnn = EGCN()
        
    def forward(self, x):
        # Feature extraction
        features, heatmaps = self.backbone(x)  # features: [B,512,8,8], heatmaps: [B,K,8,8]
        
        # Initial keypoint detection
        coords = soft_argmax(heatmaps)  # [B, K, 2]
        
        # Feature sampling at keypoints
        B, C, H, W = features.shape
        grid = coords.unsqueeze(2)*2-1  # Normalize to [-1,1]
        sampled_features = F.grid_sample(
            features.unsqueeze(1).expand(-1,NUM_KEYPOINTS,-1,-1,-1).reshape(B*NUM_KEYPOINTS,C,H,W),
            grid.reshape(B*NUM_KEYPOINTS,1,1,2),
            align_corners=False
        ).squeeze().view(B, NUM_KEYPOINTS, C)
        
        # Prepare graph nodes
        nodes = torch.cat([sampled_features, coords], dim=2)  # [B, K, 512+2]
        
        # Process graph
        edge_index = self._prepare_edge_index(B)
        refined_coords, cls_logits = self.gnn(nodes.view(B*NUM_KEYPOINTS,-1), edge_index)
        
        return refined_coords.view(B,NUM_KEYPOINTS,2), cls_logits.view(B,NUM_KEYPOINTS,NUM_CLASSES)
    
    def _prepare_edge_index(self, batch_size):
        return EDGE_INDEX.unsqueeze(1).repeat(1, batch_size, 1) + \
               torch.arange(batch_size, device=EDGE_INDEX.device).unsqueeze(0,2)*NUM_KEYPOINTS

# 6. Loss Function
class PoseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.reg_loss = nn.MSELoss()
        self.cls_loss = nn.CrossEntropyLoss()
        
    def forward(self, pred_coords, pred_cls, gt_coords, gt_labels):
        return self.reg_loss(pred_coords, gt_coords) + self.cls_loss(pred_cls, gt_labels)

# Example Usage
if __name__ == "__main__":
    model = PosePipeline()
    dummy_input = torch.randn(2, 3, 256, 256)  # Batch of 2 RGB images
    coords, cls = model(dummy_input)
    
    print("Coordinates shape:", coords.shape)  # [2, 16, 2]
    print("Classification shape:", cls.shape)  # [2, 16, 10]