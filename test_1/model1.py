############################################################
# full_pose_pipeline_accurate.py
# Vedansh – Optimized CNN + Enhanced GCN for Accurate 2D Pose Estimation
############################################################
import math, json, random, pathlib, time
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import CocoDetection
from tqdm import tqdm
from torch.utils.data.dataloader import default_collate
import matplotlib.pyplot as plt

# ----------- Helper to skip empty samples ------------ #
def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    return None if len(batch) == 0 else default_collate(batch)

# -------------------- Dataset Wrapper ------------------ #
class CocoKeypointDataset(CocoDetection):
    def __init__(self, root, annFile, image_size=256):
        super().__init__(root, annFile)
        root_path = pathlib.Path(root)
        self.ids = [img_id for img_id, info in self.coco.imgs.items()
                    if (root_path / info['file_name']).is_file()]
        self.image_size = image_size
        self.K = 17
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        img, ann = super().__getitem__(idx)
        person = next((a for a in ann if a.get("num_keypoints", 0) > 5), None)
        if person is None:
            return None

        keypoints = torch.as_tensor(person["keypoints"]).view(-1, 3).float()
        bbox = torch.as_tensor(person["bbox"]).float()

        if random.random() < 0.5:
            img = transforms.functional.hflip(img)
            keypoints[:, 0] = img.width - keypoints[:, 0]

        return self.tf(img), keypoints, bbox

# -------------------- CNN Backbone ------------------ #
class KeypointCNN(nn.Module):
    def __init__(self, num_kpts=17):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(2048, 256, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, num_kpts, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.heatmap_head(x)

# -------------------- Graph Operations ------------------ #
def extract_kpt_coords(heatmaps, topk=1):
    B, K, H, W = heatmaps.shape
    flat = heatmaps.view(B, K, -1)
    conf, ind = flat.max(-1)
    y, x = ind.div(W, rounding_mode='floor'), ind % W
    coords = torch.stack([x, y], dim=-1).float()
    return coords / W, conf.sigmoid()  # Normalized to [0, 1]

def build_knn_graph(coords, k=4):
    K = coords.size(0)
    dists = torch.cdist(coords, coords, p=2)
    knn = dists.argsort(dim=-1)[:, 1:k+1]
    adj = torch.zeros(K, K, device=coords.device)
    for i in range(K):
        adj[i, knn[i]] = 1
    adj = ((adj + adj.T) > 0).float()
    return adj

# -------------------- GCN Layers ------------------ #
class EGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, tau=0.1):
        super().__init__()
        self.W_self = nn.Linear(in_dim, out_dim)
        self.W_msg = nn.Linear(in_dim, out_dim, bias=False)
        self.tau = tau
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, h, coords, adj):
        B, K, D = h.shape
        dist = torch.cdist(coords, coords, p=2)
        attn = torch.exp(-dist / self.tau) * adj
        attn = attn / (attn.sum(-1, keepdim=True) + 1e-8)
        msg = torch.matmul(attn, self.W_msg(h))
        out = self.W_self(h) + msg
        return F.relu(self.bn(out.view(B*K, -1)).view(B, K, -1))

# -------------------- Full Model ------------------ #
class PoseEGCN(nn.Module):
    def __init__(self, cnn_backbone, num_kpts=17, hidden=128, stages=3):
        super().__init__()
        self.cnn = cnn_backbone
        self.embed = nn.Linear(1, hidden)
        self.gcn = nn.ModuleList([EGCNLayer(hidden, hidden) for _ in range(stages)])
        self.final = nn.Linear(hidden, 2)

    def forward(self, img):
        heatmaps = self.cnn(img)
        coords, conf = extract_kpt_coords(heatmaps)
        B, K = conf.shape
        h = self.embed(conf.unsqueeze(-1))
        out_coords = []
        for b in range(B):
            adj = build_knn_graph(coords[b])
            h_b = h[b:b+1]
            c_b = coords[b:b+1]
            for layer in self.gcn:
                h_b = layer(h_b, c_b, adj.unsqueeze(0))
            out = self.final(h_b)
            out_coords.append(out)
        offsets = torch.cat(out_coords, dim=0)
        final_coords = torch.clamp(coords * 256 + offsets, 0, 255)
        return final_coords, conf

# -------------------- Loss Functions ------------------ #
def heatmap_loss(pred_heat, gt_kpts, sigma=2):
    B, K, H, W = pred_heat.shape
    loss = 0
    for b in range(B):
        gt = torch.zeros_like(pred_heat[b])
        for k in range(K):
            x, y, v = gt_kpts[b, k]
            if v < 1: continue
            cx, cy = int(x/8), int(y/8)
            if cx<0 or cy<0 or cx>=W or cy>=H: continue
            grid_x = torch.arange(W, device=gt.device)
            grid_y = torch.arange(H, device=gt.device).unsqueeze(1)
            g = torch.exp(-((grid_x - cx)**2 + (grid_y - cy)**2)/(2*sigma**2))
            gt[k] = g
        loss += F.mse_loss(pred_heat[b], gt)
    return loss / B

# -------------------- Training and Evaluation ------------------ #
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    skipped, running_loss = 0, 0
    pbar = tqdm(loader, desc='Train')
    for step, batch in enumerate(pbar):
        if batch is None:
            skipped += 1
            continue
        img, kpts, _ = batch
        img, kpts = img.to(device), kpts.to(device)
        optimizer.zero_grad()
        pred_heat = model.cnn(img)
        coords_pred, _ = extract_kpt_coords(pred_heat)
        coords_pred = coords_pred * 256  # scaled back
        mask = (kpts[..., 2] > 0).unsqueeze(-1)
        l1 = (torch.abs(coords_pred - kpts[..., :2]) * mask).mean()
        loss = 5.0 * heatmap_loss(pred_heat, kpts) + 0.1 * l1
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.set_postfix(loss=f"{running_loss/(step+1):.4f}", skipped=skipped)

# -------------------- Validation ------------------ #
def validate(model, loader, device):
    model.eval()
    total_err, total_visible = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            img, kpts, _ = batch
            img, kpts = img.to(device), kpts.to(device)
            coords, _ = model(img)
            err = ((coords - kpts[..., :2])**2).sum(-1).sqrt()
            visible = (kpts[..., 2] > 0)
            total_err += (err * visible).sum().item()
            total_visible += visible.sum().item()
    pck_thresh = 0.05 * 256
    pck = (err < pck_thresh).float()[visible].mean().item() if total_visible else float("nan")
    return total_err / total_visible if total_visible else float("nan"), pck

# -------------------- Main ------------------ #
def main():
    coco_root = r"D:\\datasets\\coco\\val2017"
    ann_file = r"D:\\datasets\\coco\\coco2017\\annotations\\person_keypoints_val2017.json"
    train_ds = CocoKeypointDataset(coco_root, ann_file)
    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_skip_none)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PoseEGCN(KeypointCNN()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(1, 51):
        print(f"\nEpoch {epoch}")
        train_one_epoch(model, train_dl, optimizer, device)
        err, pck = validate(model, train_dl, device)
        print(f"Validation: Pixel Error = {err:.2f}, PCK@0.05 = {pck:.3f}")
        scheduler.step()

    torch.save(model.state_dict(), "pose_egcn_best.pth")

if __name__ == "__main__":
    main()