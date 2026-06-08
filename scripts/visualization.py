from PIL.Image import Image
import torch, io
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA

@torch.no_grad()
def extract_patches(encoder, image, layer='skip3', device=None):
    encoder.eval()
    if device is not None:
        image = image.to(device)
    bottleneck, skips = encoder(image)
    features = skips[-1] if layer == 'skip3' else bottleneck
    C, H, W = features.shape[1:]
    patches = features.squeeze(0).reshape(C, -1).permute(1, 0).cpu().numpy()
    return patches, (H, W)

def compute_pca_map(patches, spatial_shape):
    n_components = min(3, patches.shape[0], patches.shape[1])
    projected = PCA(n_components=n_components).fit_transform(patches)
    for j in range(projected.shape[1]):
        comp = projected[:, j]
        projected[:, j] = (comp - comp.min()) / (comp.max() - comp.min() + 1e-8)
    H, W = spatial_shape
    pca_map = projected.reshape(H, W, -1)
    if pca_map.shape[-1] < 3:
        pca_map = np.concatenate([pca_map, np.zeros((H, W, 3 - pca_map.shape[-1]))], axis=-1)
    return pca_map

def edge_magnitude(x):
    gx = cv2.Sobel(x.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)

def depth_edge_alignment(pca_map, depth_map):
    pca_gray = cv2.cvtColor((pca_map * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    valid = depth_map > 1e-3
    if valid.sum() < 10:
        return 0.0
    depth_norm = depth_map.copy().astype(np.float32)
    depth_norm[valid] = (depth_norm[valid] - depth_norm[valid].min()) / (depth_norm[valid].max() - depth_norm[valid].min() + 1e-8)
    p_edges = edge_magnitude(pca_gray)
    d_edges = edge_magnitude(depth_norm)
    p = p_edges[valid].reshape(-1)
    d = d_edges[valid].reshape(-1)
    if p.std() < 1e-8 or d.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(p, d)[0, 1])

def pca_visualize(encoder, images, depth_maps=None, n=4, title='', device=None, return_fig=False):
    n = min(n, images.shape[0])
    cols = 4 if depth_maps is not None else 2
    fig, axes = plt.subplots(n, cols, figsize=(16, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    mn, sd = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    scores = []
    for i in range(n):
        img = (images[i].cpu().numpy().transpose(1, 2, 0) * sd + mn).clip(0, 1)
        # Move tensor to correct device before passing to encoder
        batch_image = images[i:i+1]
        if device is not None:
            batch_image = batch_image.to(device)
        patches, (Hf, Wf) = extract_patches(encoder, batch_image, device=device)
        pm = compute_pca_map(patches, (Hf, Wf))
        pm = cv2.resize(pm, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        axes[i, 0].imshow(img); axes[i, 0].set_title('Input'); axes[i, 0].axis('off')
        axes[i, 1].imshow(pm); axes[i, 1].set_title('PCA patch map'); axes[i, 1].axis('off')
        if depth_maps is not None:
            # With:
            depth = depth_maps[i, 0].detach().cpu().numpy()
            # Stretch to [0,1] for visualization
            vmin, vmax = depth.min(), depth.max()
            if vmax > vmin:
                depth_viz = (depth - vmin) / (vmax - vmin)
            else:
                depth_viz = depth
            axes[i, 2].imshow(depth_viz, cmap='viridis')
            axes[i, 2].set_title(f'Depth [{vmin:.3f}, {vmax:.3f}]')
            axes[i, 3].axis('off')
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if return_fig:
        plt.close(fig)
        return fig, scores
    else:
        plt.show()
        plt.close(fig)
        return None, scores
    
# Helper to convert matplotlib figure to PIL Image for W&B logging
def fig_to_image(fig):
    """Convert a matplotlib figure to a PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    img.load()  # Force load to memory before buffer goes out of scope
    return img

def _denormalize_rgb(rgb_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=rgb_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=rgb_tensor.device).view(3, 1, 1)
    return (rgb_tensor * std + mean).clamp(0, 1)


def _visualize_depth_prediction(model, rgb, depth, device, title="Depth Prediction"):
    model.eval()
    rgb_batch = rgb.unsqueeze(0).to(device)
    depth_batch = depth.unsqueeze(0).to(device)

    with torch.inference_mode():
        pred = model(rgb_batch)

    pred_map = pred.squeeze().detach().cpu().numpy()
    gt_map = depth_batch.squeeze().detach().cpu().numpy()
    rgb_vis = _denormalize_rgb(rgb).permute(1, 2, 0).detach().cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].imshow(rgb_vis)
    axs[0].set_title("Input Image")
    axs[0].axis("off")

    im1 = axs[1].imshow(pred_map, cmap="inferno")
    axs[1].set_title("Predicted Depth")
    axs[1].axis("off")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    im2 = axs[2].imshow(gt_map, cmap="inferno")
    axs[2].set_title("Ground Truth Depth")
    axs[2].axis("off")
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()
    return fig