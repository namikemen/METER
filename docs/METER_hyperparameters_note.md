# METER — Hyperparameters & Experimental Settings

**Source:** Papa, Russo & Amerini, *METER: A Mobile Vision Transformer Architecture for Monocular Depth Estimation*, IEEE TCSVT, Vol. 33, No. 10, October 2023.  
**DOI:** [10.1109/TCSVT.2023.3260310](https://doi.org/10.1109/TCSVT.2023.3260310)  
**Code (paper):** https://github.com/lorenzopapa5/METER

This note collects every training, architecture, loss, augmentation, dataset, and evaluation setting stated in the paper (Section III–IV and ablations in Section V).

---

## 1. Model variants (METER S / XS / XXS)

| Variant | Trainable params | MAC (reported) | Role (paper) |
|--------|------------------|----------------|--------------|
| **METER S** | **3.29 M** | — | Best accuracy |
| **METER XS** | **1.45 M** | on par with SPEED | Speed vs. error trade-off |
| **METER XXS** | **0.71 M** | **0.186 G** | Fastest (up to **25.8 fps** on Jetson TX1) |

**Runtime (Jetson TX1, single image, test set average):**

| Model | fps |
|-------|-----|
| SPEED [7] | 30.9 |
| FastDepth [8] | 18.8 |
| METER S | 16.3 |
| METER XS | 18.3 |
| METER XXS | 25.8 |

**Memory:** METER variants use **< 2.1 GB** RAM on 4 GB Jetson boards (OS reserves part of RAM).

---

## 2. Architecture hyperparameters

### 2.1 High-level design

- **Structure:** Hybrid **encoder–decoder** (ViT-style encoder + fully convolutional decoder).
- **Encoder base:** Modified **MobileViT** [9]; expensive MobileViT blocks replaced by **METER blocks** (fewer cascaded transformers).
- **Activation:** **ReLU** (SiLU from MobileViT replaced; ~3% of encoder gain attributed to ReLU in ablation).
- **Encoder bottleneck:** Output feature channels at **C6 halved** vs. original MobileViT.
- **Decoder:** 3 cascaded **upsampling blocks** + conv at input/output; **transposed conv** `3×3`, stride 2 (×2 resolution); **skip connections** from encoder; **separable conv** `3×3` in upsampling blocks.

### 2.2 METER block (encoder)

Per block (vs. MobileViT’s four convs + multiple transformers):

- 2× **Conv block:** `3×3` conv + pointwise (`1×1`).
- **1×** transformer block (unfold → attention → fold).
- **1×1** conv; **concat** block input with transformer output before `1×1` conv.

### 2.3 Channel widths \(C_i\) — Table I

The paper refers to **Table I** for per-stage channel counts. The official repo uses these lists (11 stage widths + final):

| Stage index | METER **S** | METER **XS** | METER **XXS** |
|-------------|------------|--------------|---------------|
| C₁…C₁₁ | 16, 32, 64, 64, 96, 96, 128, 128, 160, 160, **320** | 16, 32, 48, 48, 64, 64, 80, 80, 96, 96, **192** | 16, 16, 24, 24, 48, 48, 64, 64, 80, 80, **160** |

### 2.4 Transformer / MobileNet-style encoder details

| Hyperparameter | METER S | METER XS | METER XXS |
|----------------|---------|----------|-----------|
| Transformer token dims | **144, 192, 240** | **96, 120, 144** | **64, 80, 96** |
| MV2 expansion ratio | **4** | **4** | **2** |
| Transformer depth per block **L** | **1, 1, 1** | **1, 1, 1** | **1, 1, 1** |
| Patch size (unfold/fold) | **(2, 2)** | **(2, 2)** | **(2, 2)** |
| Conv kernel (local blocks) | **3×3** | **3×3** | **3×3** |
| Attention **heads** | **4** | **4** | **4** |
| Attention **dim_head** | **8** | **8** | **8** |
| MLP dim (per block) | **2× / 4×** token dim | same | same |
| Transformer **dropout** | **0** | **0** | **0** |

---

## 3. Balanced loss function (BLF)

**Total loss (Eq. 1):**

\[
\mathcal{L} = \mathcal{L}_{\text{depth}} + \lambda_1 \mathcal{L}_{\text{grad}} + \lambda_2 \mathcal{L}_{\text{norm}} + \lambda_3 \mathcal{L}_{\text{SSIM}}
\]

| Component | Definition (paper) | Role |
|-----------|-------------------|------|
| \(\mathcal{L}_{\text{depth}}\) | Mean **L1** per pixel (Eq. 2) | Global reconstruction |
| \(\mathcal{L}_{\text{grad}}\) | **Sobel** on \(\|y_i - \hat{y}_i\|\) (Eq. 3) | Edges / boundaries |
| \(\mathcal{L}_{\text{norm}}\) | **1 − cosine similarity** of surface normals (Eq. 4) | Fine geometry |
| \(\mathcal{L}_{\text{SSIM}}\) | **1 − SSIM** (Eq. 5) | Structural similarity |

**Loss weights (Section IV-A):**

| Symbol | Value | Notes |
|--------|-------|--------|
| \(\lambda_1\) | **0.5** | Fixed |
| \(\lambda_2\) | **{1, 10, 100}** | By depth unit: **m / dm / cm** |
| \(\lambda_3\) | **{1, 10, 100}** | Same as \(\lambda_2\) |

**SSIM:** \(C_1 = (0.01 L)^2\), \(C_2 = (0.03 L)^2\) with \(L\) = depth dynamic range.

---

## 4. Data augmentation

**Global probability:** **0.5** per random transform (Section IV-A).

### 4.1 Default policy ([16])

- Vertical **flip**, **mirror**, **random crop**, **channel swap** (RGB + aligned depth).

### 4.2 Shifting strategy

**C-shift (RGB):** \(\text{rgb}_{gb} = \beta \cdot (\text{rgb}_{un})^{\gamma}\); then \(\text{rgb}_{aug} = \text{rgb}_{gb} \odot (I \cdot \eta)\).

| Parameter | Range |
|-----------|--------|
| \(\beta\) (brightness) | **[0.9, 1.1]** |
| \(\gamma\) (gamma) | **[0.9, 1.1]** |
| \(\eta\) (color scale) | **[0.9, 1.1]** |

**D-shift (depth):** uniform offset on full map.

| Dataset | Range |
|---------|--------|
| **NYU Depth v2** | **[-10, +10] cm** |
| **KITTI** | **[-10, +10] dm** |

---

## 5. Training hyperparameters

| Setting | Value |
|---------|--------|
| Framework | **PyTorch** |
| Init | **Random**, train **from scratch** |
| Optimizer | **AdamW** |
| \(\beta_1\) | **0.9** |
| \(\beta_2\) | **0.999** |
| Weight decay | **0.01** |
| Learning rate | **0.001** |
| LR schedule | **×0.1 every 20 epochs** |
| Epochs | **60** |
| Batch size | **128** |

---

## 6. Datasets

### NYU Depth v2

| Item | Value |
|------|--------|
| Native resolution | 640 × 480 |
| Max depth | **10 m** |
| Train subset | **50 K** (of 120 K) |
| Test | **654** |
| Input size | **256 × 192** |

### KITTI

| Item | Value |
|------|--------|
| Native RGB | 1241 × 376 |
| Max depth | **80 m** |
| Split | **Eigen** (~23 K train / 697 test) |
| Input size | **636 × 192** |
| Eval | **Cropped** to lidar-valid region |

---

## 7. Evaluation

| Metric | Notes |
|--------|--------|
| **RMSE** | meters |
| **REL** | mean \|y−ŷ\|/y |
| **δ₁** | threshold **thr = 1.25** |
| **MAC** | reported in **G** |
| **fps** | Jetson TX1 / Nano, dataset average |

**Hardware:** Jetson TX1 (4 GB, 10 W, 256-core Maxwell); Jetson Nano (4 GB, 5 W, 128-core Maxwell).

---

## 8. Quick training recipe

```text
AdamW(lr=1e-3, betas=(0.9, 0.999), wd=0.01); lr×0.1 @ epochs 20,40
epochs=60; batch=128
loss: L1 + 0.5·L_grad + λ2·L_norm + λ3·L_SSIM  (λ2,λ3 ∈ {1,10,100} by unit)
aug: p=0.5; β,γ,η∈[0.9,1.1]; depth shift ±10 cm (NYU) or dm (KITTI)
```

---

## 9. This repo alignment

- Channels/dims: `architecture.py` (`mobilevit_s`, `xs`, `xxs`).
- `globals.py`: `RGB_img_res = (3, 192, 256)`; augmentation **p = 0.5**.
- `meter_loss.py`: **λ₁=0.5, λ₂=λ₃=100** with depth in **cm** — rescale λ₂, λ₃ for KITTI (dm) per paper.

---

*From `docs/METER_A_Mobile_Vision_Transformer_Architecture_for_Monocular_Depth_Estimation.pdf`.*
