import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

def pad_depth_to(t: torch.Tensor, target_d: int) -> torch.Tensor:
    d = t.shape[1]
    if d == target_d:
        return t
    pad_back = target_d - d
    # pad: (W_left, W_right, H_left, H_right, D_left, D_right)
    return F.pad(t, (0, 0, 0, 0, 0, pad_back), mode='constant', value=0.0)

def collate_variable_depth(batch):
    xs, ys = zip(*batch)                     # xs: (1,D,H,W)
    max_d = max(x.shape[1] for x in xs)
    xs_padded = [pad_depth_to(x, max_d) for x in xs]
    x_batch = torch.stack(xs_padded, dim=0)  # (B,1,maxD,H,W)
    y_batch = torch.stack(ys, dim=0)
    return x_batch, y_batch

    # === Вспомогательная функция: батчевый ресэмплинг на GPU (правка №1) ===
def gpu_resample(x):
    # x: (B, 1, D, H, W) на device
    return F.interpolate(
        x, size=(TARGET_SLICES, TARGET_HW, TARGET_HW),
        mode='trilinear', align_corners=False
    )
# === Глобальные целевые размеры для интерполяции (используются на GPU) ===
TARGET_SLICES = 64
TARGET_HW = 512

def preprocess_scan(scan):
    """
    УПРОЩЕНО: только нормализация в [0,1], БЕЗ интерполяции.
    Интерполяция теперь выполняется пакетно на GPU в тренировочном/валид. цикле.
    """
    scan = scan.astype(np.float32)
    scan -= scan.min()
    scan /= (scan.max() + 1e-5)
    return torch.from_numpy(scan)  # (D, H, W)

import torch
import torch.nn as nn
import torch.nn.functional as F

# SE блок для 3D
class SE3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, d, h, w = x.shape
        y = x.view(b, c, -1).mean(dim=2)  # (b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        y = y.view(b, c, 1, 1, 1)
        return x * y

# Residual блок с SE
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

# Cross-Slice Attention (CSA)
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

# CSA-Net 2.5D
class CSANet2_5D(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, d_model=128, num_heads=8, num_layers=2):
        super().__init__()
        # 2.5D CNN Backbone
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
        # x: (B, 1, D, H, W)
        x = self.enc1(x); x = self.pool1(x)
        x = self.enc2(x); x = self.pool2(x)
        x = self.enc3(x); x = self.pool3(x)
        x = self.enc4(x)  # (B, d_model, D', H', W')

        # усредняем по H,W → токены = срезы
        x = x.mean(dim=(3, 4))        # (B, d_model, D')
        x = x.permute(0, 2, 1)        # (B, D', d_model)

        # Добавляем CLS-токен
        cls_tokens = self.cls_token.expand(x.size(0), 1, -1)  # (B, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)                 # (B, 1 + D', d_model)

        # CSA
        for layer in self.csa_layers:
            x = layer(x)

        x = self.norm(x)
        cls_final = x[:, 0, :]        # (B, d_model)
        logits = self.head(cls_final) # (B, out_channels)
        return logits


class LungCTDataset(Dataset):
    def __init__(self, df, data_dir, target_slices=TARGET_SLICES, transform=None):
        self.df = df
        self.data_dir = data_dir
        self.target_slices = target_slices
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.normpath(os.path.join(self.data_dir, row['file']))
        try:
            with np.load(file_path, allow_pickle=True) as npz:
                scan_key = list(npz.keys())[0]
                scan = np.array(npz[scan_key], dtype=np.float32)

            if scan.ndim != 3:
                raise ValueError(f"Expected 3D array, got shape {scan.shape}")

            # ТОЛЬКО нормализация, БЕЗ интерполяции (ресэмпл будет на GPU для батча)
            scan = preprocess_scan(scan)          # (D, H, W) -> Tensor
            scan = scan.unsqueeze(0).float()      # (1, D, H, W)

            label = torch.tensor(int(row['label']), dtype=torch.long)
            return scan, label

        except Exception as e:
            print(f"[DATA ERROR] Failed to load {file_path}: {e}")
            dummy_scan = torch.zeros((1, self.target_slices, TARGET_HW, TARGET_HW), dtype=torch.float32)
            dummy_label = torch.tensor(0, dtype=torch.long)
            return dummy_scan, dummy_label

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Weighted Cross-Entropy Loss ---
def get_weighted_ce(weights=None):
    return nn.CrossEntropyLoss(weight=weights)

# --- Unified Focal Loss (устойчивая версия) ---
class UnifiedFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        logp = F.log_softmax(logits, dim=1)
        nll = F.nll_loss(logp, targets, reduction='none', weight=self.alpha)
        p = torch.exp(-nll)
        loss = ((1 - p) ** self.gamma) * nll
        return loss.mean()

# --- Combined Loss ---
class CombinedLoss(nn.Module):
    def __init__(self, weights=None, gamma=2.0, alpha=None, ce_weight=0.5, focal_weight=0.5):
        super().__init__()
        self.ce_loss = get_weighted_ce(weights)
        self.focal_loss = UnifiedFocalLoss(gamma=gamma, alpha=alpha)
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight

        print("=== CombinedLoss initialized ===")
        if weights is not None:
            print("CE class weights (mean≈1):", weights)
        if alpha is not None:
            print("Focal alpha weights:", alpha)

    def forward(self, logits, targets):
        loss_ce = self.ce_loss(logits, targets)
        loss_focal = self.focal_loss(logits, targets)
        loss_combined = self.ce_weight * loss_ce + self.focal_weight * loss_focal
        return loss_combined, loss_ce, loss_focal

# ===== Метрики предсказаний (гистограмма, уверенность, энтропия, per-class acc) =====
# ===== Метрики предсказаний (GPU-friendly) =====
def init_pred_stats(num_classes: int, device=None):
    device = device or torch.device("cpu")
    return {
        "n": 0,
        "correct": 0,
        "hist": torch.zeros(num_classes, dtype=torch.long, device=device),
        "per_class_corr": torch.zeros(num_classes, dtype=torch.long, device=device),
        "per_class_total": torch.zeros(num_classes, dtype=torch.long, device=device),
        "conf_sum": torch.tensor(0.0, device=device),
        "ent_sum":  torch.tensor(0.0, device=device),
    }

@torch.no_grad()
def update_pred_stats(stats: dict, logits: torch.Tensor, targets: torch.Tensor, num_classes: int):
    # всё на ОДНОМ device (GPU), без .cpu() в батч-цикле
    probs = torch.softmax(logits, dim=1)
    logp  = torch.log_softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    ent = -(probs * logp).sum(dim=1)

    stats["n"] += targets.numel()
    stats["correct"] += (pred == targets).sum().item()

    stats["hist"] += torch.bincount(pred, minlength=num_classes)
    stats["per_class_total"] += torch.bincount(targets, minlength=num_classes)

    pc_mask = (pred == targets)
    stats["per_class_corr"] += torch.bincount(targets[pc_mask], minlength=num_classes)

    stats["conf_sum"] += conf.sum()
    stats["ent_sum"]  += ent.sum()

def finalize_pred_stats(stats: dict):
    # только здесь переносим на CPU
    n = max(stats["n"], 1)
    mean_conf = (stats["conf_sum"] / n).item()
    mean_ent  = (stats["ent_sum"]  / n).item()
    overall_acc = 100.0 * stats["correct"] / n

    per_class_total = stats["per_class_total"].clamp_min(1)
    per_class_acc = (stats["per_class_corr"].float() / per_class_total.float()) * 100.0

    return {
        "overall_acc": overall_acc,
        "hist": stats["hist"].detach().cpu(),
        "mean_conf": mean_conf,
        "mean_ent": mean_ent,
        "per_class_acc": per_class_acc.detach().cpu(),
    }

def pretty_print_pred_stats(name: str, stats_dict: dict, class_names=None, log_fn=print):
    hist = stats_dict["hist"].numpy()
    total = hist.sum()
    distr = (hist / max(total, 1) * 100.0)
    if class_names is None:
        class_names = [f"class_{i}" for i in range(len(hist))]

    bar = " | ".join([f"{cls}: {int(cnt)} ({p:.1f}%)"
                      for cls, cnt, p in zip(class_names, hist, distr)])
    log_fn(
        f"[{name}] acc={stats_dict['overall_acc']:.2f}% | "
        f"mean_conf={stats_dict['mean_conf']:.3f} | mean_entropy={stats_dict['mean_ent']:.3f}\n"
        f"[{name}] pred_hist: {bar}\n"
        f"[{name}] per-class acc: " +
        " | ".join([f"{cls}: {a:.1f}%" for cls, a in zip(class_names, stats_dict["per_class_acc"].numpy())])
    )


def main():
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from sklearn.model_selection import train_test_split
    import sys
    import os
    import logging
    import torch.backends.cudnn as cudnn
    # === TF32 ===
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cudnn.benchmark = True

    # === Настройка логирования ===
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    log_filename = f"{script_name}.log"

    logging.basicConfig(
        filename=log_filename,
        filemode='w',
        format='%(asctime)s | %(levelname)s | %(message)s',
        level=logging.INFO
    )

    def log(msg):
        print(msg)
        logging.info(msg)

    # === Логирование всего кода скрипта в начале ===
    try:
        with open(__file__, 'r', encoding='utf-8') as f:
            script_code = f.read()
        logging.info("=== START OF SCRIPT CODE ===")
        logging.info("\n" + script_code)
        logging.info("=== END OF SCRIPT CODE ===\n")
    except Exception as e:
        log(f"[LOGGING ERROR] Cannot read script code: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Используемое устройство:", device)

    # ✅ AMP включаем только на CUDA
    USE_AMP = (device.type == "cuda")
    scaler = torch.amp.GradScaler(enabled=USE_AMP)

    def amp_autocast():
        # единое место: без FutureWarning
        return torch.amp.autocast("cuda", dtype=torch.float16, enabled=USE_AMP)


    data_dir = r"C:/Users/Admin/PycharmProjects/PythonProject2/Processed_old"
    csv_file = "C:/Users/Admin/PycharmProjects/PythonProject2/Processed_old/labels.csv"  # твой CSV с колонками ['patient_path','label']

    df = pd.read_csv(csv_file)
    print(df.head())
    print("CSV загружен")

    df_train, df_temp = train_test_split(df, stratify=df["label"], test_size=0.2, random_state=42)
    df_val, df_test = train_test_split(df_temp, stratify=df_temp["label"], test_size=0.1, random_state=42)
    print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    sys.stdout.flush()


    train_dataset = LungCTDataset(df_train, data_dir, target_slices=TARGET_SLICES)
    val_dataset = LungCTDataset(df_val, data_dir, target_slices=TARGET_SLICES)
    test_dataset = LungCTDataset(df_test, data_dir, target_slices=TARGET_SLICES)

    # === Тюнинг DataLoader (правка №2) ===
    batch_size = 3
    num_workers = min(8, os.cpu_count() - 2 if os.cpu_count() else 4)
    prefetch_factor = 2

    from torch.utils.data import WeightedRandomSampler

    # частоты классов в train
    train_label_counts = df_train['label'].value_counts().sort_index()
    num_classes = int(train_label_counts.index.max() + 1)

    # веса классов = 1 / частота
    class_weights_sampler = 1.0 / (train_label_counts.values + 1e-6)

    # веса для каждого сэмпла
    sample_weights = df_train['label'].map(lambda y: class_weights_sampler[int(y)]).values
    sample_weights = torch.tensor(sample_weights, dtype=torch.double)

    # СДЕЛАЕМ БАЛАНСИРОВАННУЮ ДЛИНУ ЭПОХИ:
    max_count = int(train_label_counts.max())
    epoch_len = max_count * num_classes  # одинаковое ожидаемое число на класс за эпоху

    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=epoch_len,  # <-- вместо len(sample_weights)
        replacement=True
    )

    pin_mem = (device.type == "cuda")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_variable_depth,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_mem,
        persistent_workers=True, prefetch_factor=prefetch_factor,
        collate_fn=collate_variable_depth,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_mem,
        persistent_workers=True, prefetch_factor=prefetch_factor,
        collate_fn=collate_variable_depth,
    )

    # Проверим распределение меток и допустимый диапазон
    print("Label distribution:\n", df["label"].value_counts().sort_index())
    num_classes = int(df['label'].max()) + 1
    assert df["label"].min() >= 0 and df["label"].max() < num_classes, \
        f"Labels must be in [0, {num_classes - 1}]"
    class_names = [str(c) for c in range(num_classes)]

    # === Модель / оптимизатор / шедулер ===
    model = CSANet2_5D(out_channels=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )

    def class_balanced_weights(counts, beta=0.999):  # counts: torch.float32 [C] на device
        eff_num = 1.0 - torch.pow(beta, counts)
        w = (1.0 - beta) / eff_num.clamp_min(1e-6)
        return w * (w.numel() / w.sum())  # нормируем так, чтобы среднее ≈ 1

    class_counts = torch.tensor(
        df_train['label'].value_counts().sort_index().values,
        dtype=torch.float32, device=device
    )
    cb_w = class_balanced_weights(class_counts, beta=0.999)

    criterion = CombinedLoss(
        weights=cb_w,  # -> CE
        alpha=cb_w,  # -> Focal
        gamma=1.5,  # начни с 1.5; если класс всё ещё «тонет», подними до 2.0
        ce_weight=0.5,
        focal_weight=0.5
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_ce": [],
        "val_focal": [],
        "val_acc": [],
    }

    # === Функция оценки модели ===
    from sklearn.metrics import confusion_matrix, classification_report
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch

    def evaluate(model, dataloader, device, class_names=None):
        model.eval()
        correct, total = 0, 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                x = gpu_resample(x)

                with amp_autocast():
                    out = model(x)

                _, pred = out.max(1)
                total += y.size(0)
                correct += (pred == y).sum().item()
                all_preds.append(pred.cpu())
                all_labels.append(y.cpu())

        acc = 100 * correct / total
        log(f"Test Accuracy: {acc:.2f}%")

        # --- Confusion Matrix и Classification Report ---
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)

        log("\nClassification Report:\n" + report)

        # Отрисовка confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

        return acc

    # === Оптимизатор и функция потерь ===
    # AdamW + weight decay для лучшей регуляризации
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)


    # --- Автоматический расчет и НОРМАЛИЗАЦИЯ весов классов для CE ---
    class_counts = df_train['label'].value_counts().sort_index().values  # [N0, N1, ...]
    class_counts = torch.tensor(class_counts, dtype=torch.float32, device=device)

    ce_weights = 1.0 / (class_counts + 1e-6)  # обратная пропорция
    # нормализуем так, чтобы среднее было ≈ 1 (масштаб CE будет сопоставим с Focal)
    ce_weights = ce_weights * (ce_weights.numel() / ce_weights.sum())

    # === ВКЛЮЧЕНИЕ FOCAL ПОСЛЕ N ЭПОХ ===
    EPOCH_FOCAL_START = 10  # первые 10 эпох — без Focal
    best_val_acc = 0.0

    class_names = [str(c) for c in range(num_classes)]

    # === Цикл обучения ===
    NUM_EPOCHS = 400
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        running_ce = 0.0
        running_focal = 0.0
        correct = 0
        total = 0
        # === Статистика предсказаний на train ===
        stats_train = init_pred_stats(num_classes, device=device)

        use_focal = (epoch + 1) >= EPOCH_FOCAL_START

        for i, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Пакетная интерполяция на GPU (правка №1)
            x = gpu_resample(x)
            optimizer.zero_grad(set_to_none=True)

            with amp_autocast():
                out = model(x)
                # CE всегда считаем
                loss_ce = criterion.ce_loss(out, y)
                if use_focal:
                    # Focal только после EPOCH_FOCAL_START
                    loss_focal = criterion.focal_loss(out, y)
                    loss = 0.5 * loss_ce + 0.5 * loss_focal
                else:
                    # до старта Focal пишем ноль для аккуратной статистики
                    loss_focal = torch.tensor(0.0, device=out.device)
                    loss = loss_ce

            # обновляем статистику предсказаний для train
            update_pred_stats(stats_train, out, y, num_classes)

            if USE_AMP:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            running_ce += loss_ce.item()
            running_focal += loss_focal.item()

            _, predicted = out.max(1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

            # Вывод прогресса батчей
            print(f"\rEpoch {epoch + 1}/{NUM_EPOCHS} Batch {i + 1}/{len(train_loader)} "
                  f"Loss: {running_loss / (i + 1):.4f} "
                  f"CE: {running_ce / (i + 1):.4f} "
                  f"Focal: {running_focal / (i + 1):.4f} "
                  f"Acc: {100 * correct / total:.2f}%", end="", flush=True)

        train_loss_avg = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        log(f"\n--- Epoch {epoch + 1} finished: Avg Loss: {train_loss_avg:.4f}, Avg Acc: {train_acc:.2f}% ---")

        # === Итоги по train-предсказаниям ===
        train_stats = finalize_pred_stats(stats_train)
        pretty_print_pred_stats("Train", train_stats, class_names, log_fn=log)

        # === Валидация ===
        model.eval()
        val_loss, val_ce, val_focal = 0.0, 0.0, 0.0
        val_correct = 0
        val_total = 0

        # >>> ДОБАВИТЬ: собираем валид-статы предсказаний
        stats_val = init_pred_stats(num_classes, device=device)

        with torch.no_grad():
            for i, (x_val, y_val) in enumerate(val_loader):
                x_val = x_val.to(device, non_blocking=True)
                y_val = y_val.to(device, non_blocking=True)

                x_val = gpu_resample(x_val)

                with amp_autocast():
                    out_val = model(x_val)
                    loss_val, loss_ce_val, loss_focal_val = criterion(out_val, y_val)

                # >>> ДОБАВИТЬ: обновляем валид-статистики
                update_pred_stats(stats_val, out_val, y_val, num_classes)

                val_loss += loss_val.item()
                val_ce += loss_ce_val.item()
                val_focal += loss_focal_val.item()

                _, pred_val = out_val.max(1)
                val_total += y_val.size(0)
                val_correct += (pred_val == y_val).sum().item()

        val_loss_avg = val_loss / len(val_loader)
        val_ce_avg = val_ce / len(val_loader)
        val_focal_avg = val_focal / len(val_loader)
        val_acc = 100 * val_correct / val_total

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss_avg)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss_avg)
        history["val_ce"].append(val_ce_avg)
        history["val_focal"].append(val_focal_avg)
        history["val_acc"].append(val_acc)

        scheduler.step(val_loss_avg)

        current_lr = optimizer.param_groups[0]["lr"]

        # >>> ДОБАВИТЬ: печать гистограммы предсказаний и per-class acc для валидации
        val_stats = finalize_pred_stats(stats_val)
        pretty_print_pred_stats("Val", val_stats, class_names, log_fn=log)

        log(f"Validation Loss: {val_loss_avg:.4f} | CE: {val_ce_avg:.4f} | "
            f"Focal: {val_focal_avg:.4f} | Acc: {val_acc:.2f}% | LR: {current_lr:.2e}")

        # Получаем имя скрипта без расширения
        script_name = os.path.splitext(os.path.basename(__file__))[0]

        # === Сохранение лучшей модели ===
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Формируем имя файла: <имя_скрипта>_<валид_аккуратность>.pth
            best_model_path = f"{script_name}_{best_val_acc:.2f}.pth"
            torch.save(model.state_dict(), best_model_path)
            log(f"✅ New best model saved: {best_model_path}")

    # === Тестирование ===
    log("\n=== Final Evaluation on Test Set ===")
    class_names = [str(c) for c in sorted(df['label'].unique())]
    evaluate(model, test_loader, device, class_names)

    import matplotlib.pyplot as plt

    # общая функция сглаживания (по желанию)
    def smooth(y, k=1):
        if k <= 1 or k > len(y):
            return y
        import numpy as np
        y = np.array(y, dtype=float)
        kernel = np.ones(k) / k
        return np.convolve(y, kernel, mode="valid")

    script_name = os.path.splitext(os.path.basename(__file__))[0]
    png_path = f"{script_name}_training_curves.png"

    epochs = history["epoch"]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.show()

    log(f"📈 Saved training curves to: {png_path}")


if __name__ == "__main__":
    main()