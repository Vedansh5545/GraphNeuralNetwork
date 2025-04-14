import torch
from PIL import Image
import matplotlib.pyplot as plt
from full_pose_pipeline_accurate import PoseEGCN, KeypointCNN, extract_kpt_coords

# --- Optimized PIL to Tensor (no NumPy or torchvision) ---
def pil_to_tensor(image_pil):
    image_pil = image_pil.resize((256, 256)).convert("RGB")
    r, g, b = image_pil.split()
    img_tensor = torch.stack([
        torch.ByteTensor(r.getdata()).reshape(256, 256),
        torch.ByteTensor(g.getdata()).reshape(256, 256),
        torch.ByteTensor(b.getdata()).reshape(256, 256)
    ], dim=0).float() / 255.0

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    return ((img_tensor - mean) / std).unsqueeze(0)  # B×C×H×W

def load_model(weights_path, device):
    model = PoseEGCN(KeypointCNN()).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def run_inference(model, img_tensor):
    with torch.no_grad():
        heatmaps = model.cnn(img_tensor)
        coords, conf = extract_kpt_coords(heatmaps)
        coords_scaled = coords[0] * 256  # Normalize to image size
        return coords_scaled.cpu().tolist()

def visualize(img_path, coords):
    image = Image.open(img_path).resize((256, 256))
    fig, ax = plt.subplots()
    ax.imshow(image)
    for x, y in coords:
        if 0 <= x < 256 and 0 <= y < 256:
            ax.plot(x, y, 'ro', markersize=4)
    plt.axis("off")
    plt.title("Predicted Keypoints")
    plt.tight_layout()
    plt.show()

def main():
    image_path = "A_photograph_of_a_fit_woman_with_light_to_medium_s.png"
    weights = "pose_egcn_best.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    model = load_model(weights, device)

    print("Processing image safely...")
    image = Image.open(image_path)
    img_tensor = pil_to_tensor(image).to(device)

    print("Running inference...")
    coords = run_inference(model, img_tensor)
    print("Keypoints:", coords)

    print("Visualizing...")
    visualize(image_path, coords)

if __name__ == "__main__":
    main()
