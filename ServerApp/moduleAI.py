import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pydicom
import pandas as pd

# ================= Функции =================
def preprocess_scan(scan):
    scan = scan.astype(np.float32)
    scan -= scan.min()
    scan /= (scan.max() + 1e-5)
    return torch.from_numpy(scan)

# ================= Dataset =================
class InferenceDataset(Dataset):
    def __init__(self, dicom_files):
        self.dicom_files = dicom_files

    def __len__(self):
        return 1  # одна серия

    def __getitem__(self, idx):
        scan = [pydicom.dcmread(f).pixel_array for f in self.dicom_files]
        scan = np.stack(scan, axis=0).astype(np.float32)  # (D,H,W)

        # ===== Выбираем ровно 64 среза =====
        if scan.shape[0] > 64:
            indices = np.linspace(0, scan.shape[0]-1, 64).astype(int)
            scan = scan[indices]
        elif scan.shape[0] < 64:
            pad_count = 64 - scan.shape[0]
            last_slice = scan[-1:]
            scan = np.concatenate([scan] + [last_slice]*pad_count, axis=0)

        scan_tensor = preprocess_scan(scan)  # (D,H,W)
        scan_tensor = scan_tensor.unsqueeze(0)  # (1,D,H,W)
        return scan_tensor, "series_1"

# ================= Модель =================
class SE3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, d, h, w = x.shape
        y = x.view(b, c, -1).mean(dim=2)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        y = y.view(b, c, 1, 1, 1)
        return x * y

class ResidualBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, use_se=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
        )
        self.skip = nn.Conv3d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.use_se = use_se
        if use_se:
            self.se = SE3D(out_ch)

    def forward(self, x):
        out = self.conv(x)
        out = out + self.skip(x)
        out = self.relu(out)
        if self.use_se:
            out = self.se(out)
        return out

class CSABlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x2 = self.norm1(x)
        attn_out, _ = self.attn(x2, x2, x2)
        x = x + attn_out
        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        return x

class CSANet2_5D(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, d_model=128, num_heads=8, num_layers=2):
        super().__init__()
        self.enc1 = ResidualBlock3D(in_channels, 16)
        self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.enc2 = ResidualBlock3D(16, 32)
        self.pool2 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.enc3 = ResidualBlock3D(32, 48)
        self.pool3 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.enc4 = ResidualBlock3D(48, d_model)

        self.csa_layers = nn.ModuleList([CSABlock(d_model, num_heads=num_heads) for _ in range(num_layers)])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model // 2, out_channels)
        )

    def forward(self, x):
        x = self.enc1(x); x = self.pool1(x)
        x = self.enc2(x); x = self.pool2(x)
        x = self.enc3(x); x = self.pool3(x)
        x = self.enc4(x)
        x = x.mean(dim=(3,4))   # (B,d_model,D')
        x = x.permute(0,2,1)    # (B,D',d_model)
        cls_tokens = self.cls_token.expand(x.size(0),1,-1)
        x = torch.cat([cls_tokens,x], dim=1)
        for layer in self.csa_layers:
            x = layer(x)
        x = self.norm(x)
        cls_final = x[:,0,:]
        logits = self.head(cls_final)
        return logits

# ================= Inference =================
def run_inference(model, dataloader, device):
    model.eval()
    results = []
    with torch.no_grad():
        for x_batch, series_name in dataloader:
            x_batch = x_batch.to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(x_batch)
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
            for name, pred, prob in zip(series_name, preds.cpu().numpy(), probs.cpu().numpy()):
                results.append({
                    'series_name': name,
                    'pred_class': int(pred),
                    **{f'class_{i}_prob': float(p) for i,p in enumerate(prob)}
                })
    return results

# ================= API функции модуля =================
def infer_folder(dicom_root: str, model_path: str, device=None):
    """
    Выполняет инференс серии DICOM-файлов в папке.
    Возвращает pandas.DataFrame с результатами.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dicom_files = [os.path.join(dicom_root, f) for f in os.listdir(dicom_root)
                   if f.lower().endswith(".dcm")]
    if len(dicom_files) == 0:
        raise ValueError(f"❌ В папке {dicom_root} нет DICOM-файлов!")

    dataset = InferenceDataset(dicom_files)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = CSANet2_5D().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    results = run_inference(model, dataloader, device)
    return pd.DataFrame(results)
