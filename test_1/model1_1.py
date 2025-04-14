############################################################
# pose_estimator.py
# Best-in-Class 2D Human Pose Estimation with HRNet + Graph Transformer
############################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.ops import deform_conv2d
import numpy as np
import json
import cv2
from scipy.spatial import cKDTree
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_

# -------------------- Hyperparameters -------------------- #
CFG = {
    "image_size": 512,
    "heatmap_size": 128,
    "num_joints": 17,
    "backbone": "hrnet_w48",
    "hidden_dim": 256,
    "num_encoder_layers": 4,
    "num_heads": 8,
    "graph_topk": 5,
    "graph_radius": 0.1,
    "batch_size": 16,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "max_epochs": 300,
    "flip_prob": 0.5,
    "rotate_limit": 30,
    "scale_limit": 0.2,
    "blur_prob": 0.2,
    "temperature": 0.07
}

# -------------------- Data Pipeline ----------------------- #
class CocoPoseDataset(Dataset):
    def __init__(self, root, ann_file, is_train=True):
        self.root = root
        self.coco = self._load_annotations(ann_file)
        self.ids = self._filter_valid_ids()
        self.is_train = is_train
        self.transform = self._build_transform()

    def _load_annotations(self, ann_file):
        with open(ann_file, 'r') as f:
            return json.load(f)

    def _filter_valid_ids(self):
        valid_ids = []
        for img in self.coco['images']:
            anns = [a for a in self.coco['annotations'] 
                    if a['image_id'] == img['id'] and a['num_keypoints'] > 0]
            if len(anns) > 0:
                valid_ids.append(img['id'])
        return valid_ids

    def _build_transform(self):
        if self.is_train:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(CFG['flip_prob']),
                transforms.RandomAffine(
                    degrees=CFG['rotate_limit'],
                    scale=(1-CFG['scale_limit'], 1+CFG['scale_limit']),
                    fill=(0, 0, 0)),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                transforms.RandomGrayscale(0.1),
                transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
                                   ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = next(i for i in self.coco['images'] if i['id'] == img_id)
        img = cv2.imread(f"{self.root}/{img_info['file_name']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        anns = [a for a in self.coco['annotations']
                if a['image_id'] == img_id and a['num_keypoints'] > 0]
        ann = max(anns, key=lambda x: x['area'])  # Select largest person
        
        keypoints = np.array(ann['keypoints']).reshape(-1, 3)
        keypoints[:, :2] = self._scale_coords(keypoints[:, :2], img.shape[:2])
        valid = keypoints[:, 2] > 0  # COCO visibility: 0=unlabeled, 1=occluded, 2=visible
        
        if self.is_train:
            img, keypoints = self._augment(img, keypoints)
            
        heatmaps = self._generate_heatmaps(keypoints, valid)
        img = self.transform(img)
        
        return img, torch.FloatTensor(heatmaps), torch.FloatTensor(keypoints), torch.BoolTensor(valid)

    def _scale_coords(self, coords, orig_shape):
        scale = CFG['image_size'] / max(orig_shape)
        new_shape = [int(orig_shape[0]*scale), int(orig_shape[1]*scale)]
        coords *= scale
        offset = (CFG['image_size'] - np.array(new_shape)) // 2
        coords += offset[::-1]
        return np.clip(coords, 0, CFG['image_size']-1)

    def _generate_heatmaps(self, keypoints, valid):
        heatmaps = np.zeros((CFG['num_joints'], CFG['heatmap_size'], CFG['heatmap_size']), 
                        dtype=np.float32)
        sigma = CFG['heatmap_size'] * 0.03  # ~3px in original image
        
        for i, (x, y, v) in enumerate(keypoints):
            if not valid[i]:
                continue
                
            x_hm = x * CFG['heatmap_size'] / CFG['image_size']
            y_hm = y * CFG['heatmap_size'] / CFG['image_size']
            
            xx, yy = np.meshgrid(np.arange(CFG['heatmap_size']), 
                               np.arange(CFG['heatmap_size']))
            d2 = (xx - x_hm)**2 + (yy - y_hm)**2
            heatmaps[i] = np.exp(-d2 / (2 * sigma**2))
            
        return heatmaps

# -------------------- Backbone Network -------------------- #
class HRNet(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.hrnet_w48(pretrained=True)
        self.stem = base_model.stem
        self.stage1 = base_model.stage1
        self.stage2 = base_model.stage2
        self.stage3 = base_model.stage3
        self.stage4 = base.model.stage4
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(720, CFG['hidden_dim'], 1),  # Concatenated features
            nn.BatchNorm2d(CFG['hidden_dim']),
            nn.ReLU(True)
        )
        
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x2 = self.stage2(x)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        
        # Multi-resolution feature fusion
        x2 = F.interpolate(x2, size=x4.size()[2:], mode='bilinear')
        x3 = F.interpolate(x3, size=x4.size()[2:], mode='bilinear')
        x = torch.cat([x2, x3, x4], dim=1)
        return self.final_conv(x)

# ------------------ Graph Transformer --------------------- #
class GraphEncoderLayer(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads)
        self.drop_path1 = DropPath(0.1)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim),
            nn.Dropout(0.1)
        )
        self.drop_path2 = DropPath(0.1)
        
    def forward(self, x, adj):
        x = x + self.drop_path1(self.attn(self.norm1(x), adj))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim*3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(0.1)
        
    def forward(self, x, adj):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C//self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(adj.unsqueeze(1) == 0, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

# -------------------- Full Model -------------------------- #
class PoseEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = HRNet()
        self.joint_conv = nn.Conv2d(CFG['hidden_dim'], CFG['num_joints'], 1)
        
        # Graph initialization
        self.joint_embed = nn.Linear(2, CFG['hidden_dim'])
        self.encoder_layers = nn.ModuleList([
            GraphEncoderLayer(CFG['hidden_dim'], CFG['num_heads'])
            for _ in range(CFG['num_encoder_layers'])
        ])
        
        # Regression heads
        self.offset_head = nn.Sequential(
            nn.Linear(CFG['hidden_dim'], CFG['hidden_dim']),
            nn.ReLU(),
            nn.Linear(CFG['hidden_dim'], 2)
        )
        self.conf_head = nn.Sequential(
            nn.Linear(CFG['hidden_dim'], CFG['hidden_dim']),
            nn.ReLU(),
            nn.Linear(CFG['hidden_dim'], 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Feature extraction
        features = self.backbone(x)  # B×C×H×W
        heatmaps = self.joint_conv(features)
        
        # Initial coordinates
        B, C, H, W = heatmaps.shape
        coords, conf = self._get_max_coords(heatmaps)  # B×J×2, B×J
        
        # Build dynamic graph
        adj = self._build_graph(coords)  # B×J×J
        
        # Graph processing
        embeddings = self.joint_embed(coords)  # B×J×D
        for layer in self.encoder_layers:
            embeddings = layer(embeddings, adj)
            
        # Predict refinements
        offsets = self.offset_head(embeddings)
        conf = self.conf_head(embeddings).squeeze(-1)
        
        # Scale to original image size
        final_coords = coords * (CFG['image_size'] / CFG['heatmap_size']) + offsets
        return final_coords, conf, heatmaps

    def _get_max_coords(self, heatmaps):
        B, J, H, W = heatmaps.shape
        heatmaps = heatmaps.view(B, J, -1)
        conf, indices = heatmaps.max(dim=-1)
        y = indices // W
        x = indices % W
        return torch.stack([x, y], dim=-1).float(), conf

    def _build_graph(self, coords):
        B, J, _ = coords.shape
        adj = torch.zeros(B, J, J, device=coords.device)
        
        for b in range(B):
            # Spatial connections
            dists = torch.cdist(coords[b], coords[b])
            _, topk = torch.topk(dists, CFG['graph_topk'], dim=-1, largest=False)
            adj[b].scatter_(-1, topk, 1)
            
            # Semantic connections
            tree = cKDTree(coords[b].cpu().numpy())
            pairs = tree.query_pairs(CFG['graph_radius'] * CFG['image_size'])
            for i, j in pairs:
                adj[b, i, j] = 1
                adj[b, j, i] = 1
                
        return adj

# -------------------- Loss Function ----------------------- #
class PoseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.heatmap_loss = nn.MSELoss()
        self.offset_loss = nn.L1Loss()
        self.conf_loss = nn.BCELoss()
        
    def forward(self, pred_coords, pred_conf, pred_heat, gt_heat, gt_coords, valid):
        # Heatmap loss
        heat_loss = self.heatmap_loss(pred_heat, gt_heat)
        
        # Coordinate regression
        valid = valid.unsqueeze(-1)
        offset_loss = self.offset_loss(pred_coords[valid], gt_coords[valid])
        
        # Confidence prediction
        conf_loss = self.conf_loss(pred_conf, valid.float().squeeze(-1))
        
        # Geometric consistency
        bone_length_loss = self._bone_length_loss(pred_coords, gt_coords, valid)
        
        return (heat_loss + 0.5*offset_loss + 0.2*conf_loss + 0.1*bone_length_loss)

    def _bone_length_loss(self, pred, gt, valid):
        # Predefined COCO bone connections
        bones = [(0,1), (1,2), (2,3), (3,4), (1,5), (5,6),
                (6,7), (1,8), (8,9), (9,10), (10,11), (8,12),
                (12,13), (13,14)]
        
        loss = 0
        for (i,j) in bones:
            valid_pairs = valid[:,i] & valid[:,j]
            if valid_pairs.sum() == 0:
                continue
                
            pred_len = torch.norm(pred[:,i] - pred[:,j], dim=-1)
            gt_len = torch.norm(gt[:,i] - gt[:,j], dim=-1)
            loss += F.l1_loss(pred_len[valid_pairs], gt_len[valid_pairs])
            
        return loss / len(bones)

# -------------------- Training Utilities ------------------ #
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc="Training")
    
    for images, heatmaps, coords, valid in pbar:
        images = images.to(device)
        heatmaps = heatmaps.to(device)
        coords = coords.to(device)
        valid = valid.to(device)
        
        optimizer.zero_grad()
        pred_coords, pred_conf, pred_heat = model(images)
        
        loss = criterion(pred_coords, pred_conf, pred_heat, heatmaps, coords, valid)
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=total_loss/len(pbar))
    
    return total_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    pck = []
    with torch.no_grad():
        for images, _, coords, valid in tqdm(loader, desc="Validation"):
            images = images.to(device)
            coords = coords.to(device)
            valid = valid.to(device)
            
            pred_coords, _, _ = model(images)
            error = torch.norm(pred_coords - coords, dim=-1)
            
            # PCK@0.05 calculation
            img_size = CFG['image_size']
            threshold = 0.05 * img_size
            correct = (error < threshold) & valid
            pck.append(correct.float().mean().item())
    
    return np.mean(pck)

# -------------------- Main Training Loop ------------------ #
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data
    train_set = CocoPoseDataset(COCO_ROOT, COCO_ANN_TRAIN)
    val_set = CocoPoseDataset(COCO_ROOT, COCO_ANN_VAL, is_train=False)
    
    train_loader = DataLoader(train_set, batch_size=CFG['batch_size'], shuffle=True,
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=CFG['batch_size'], 
                           num_workers=4, pin_memory=True)
    
    # Model
    model = PoseEstimator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], 
                                weight_decay=CFG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, CFG['lr'], total_steps=CFG['max_epochs']*len(train_loader))
    criterion = PoseLoss()
    
    # Training loop
    best_pck = 0
    for epoch in range(CFG['max_epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_pck = validate(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{CFG['max_epochs']}")
        print(f"Train Loss: {train_loss:.4f} | Val PCK@0.05: {val_pck:.4f}")
        
        if val_pck > best_pck:
            best_pck = val_pck
            torch.save(model.state_dict(), f"pose_estimator_best.pth")
            print("New best model saved!")