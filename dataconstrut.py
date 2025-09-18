# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import torch
from dataclasses import dataclass
from typing import Tuple, List
from scipy.signal import stft

# =========================
# 配置区（按需修改）
# =========================
@dataclass
class Config:
    sample_rate: int = 128
    baseline_sec: int = 3          # 基线时长（秒）
    total_sec: int = 60            # 有效数据时长（秒）
    window_sec: int = 3            # 分窗大小（秒）
    step_sec: int = 2              # 分窗步长（秒）
    stft_win_sec: float = 2.0      # STFT 窗长（秒）
    stft_overlap_ratio: float = 0.5
    stft_nfft: int = None          # None → 使用 nperseg
    harmonic_order: int = 8        # 2-9 次谐波（共 8 个）
    min_band_width_hz: float = 2.0 # 频带最小带宽（Hz）
    label_dim: int = 0             # DEAP 标签维度（0:valence, 1:arousal, 2:dominance, 3:liking）
    label_threshold: float = 5.0   # DEAP 二值阈值
    out_dir: str = "./eegall/data/DEAP"   # 输出根目录
    raw_dir: str = "./DEAP/data_preprocessed_python/data_preprocessed_python/raw"  # 原始 pickle 路径

CFG = Config()

# =========================
# 工具函数
# =========================
def read_eeg_signal_from_file(filename: str):
    with open(filename, "rb") as f:
        x = pickle._Unpickler(f)
        x.encoding = "latin1"
        p = x.load()
    return p  # dict, keys: 'data' (40x40x8064?) in DEAP原数据; 这里假设外部已转成 (channels, time)

def data_calibrate(data: np.ndarray, sample_rate: int, baseline_sec: int) -> np.ndarray:
    """
    基线抵消（可选）：把 first baseline_sec 作为基线，重复拼到 total_sec 再相减。
    若不需要抵消，将最后一行注释的相减替换为直接 return normal_data。
    输入 data 形状: (channels, total_len)
    """
    fs = sample_rate
    baseline_len = baseline_sec * fs
    baseline_data, normal_data = np.split(data, [baseline_len], axis=-1)
    # 把 baseline 拉到与 normal_data 同长度（按时间重复）
    reps = int(np.ceil(normal_data.shape[-1] / baseline_data.shape[-1]))
    baseline_tiled = np.tile(baseline_data, reps)[:, :normal_data.shape[-1]]
    # 如需启用基线抵消，使用下面一行；如果不需要，就返回 normal_data
    return normal_data - baseline_tiled
    # return normal_data

def set_label(labels: np.ndarray, dim: int = 0, threshold: float = 5.0) -> int:
    """
    labels: (40, 4) for DEAP，每个 trial 的 4 维评分。
    返回二值标签（单一维度），int 0/1。
    """
    if labels.ndim == 1:
        score = labels[dim]
    else:
        score = labels[:, dim].mean() if labels.shape[0] == 40 else labels[dim]
    return int(0 if score < threshold else 1)

def data_divide(data: np.ndarray, fs: int, window_sec: int, step_sec: int) -> np.ndarray:
    """
    输入:
        data: (channels, time)
    输出:
        segments: (num_segments, channels, window_len)
    """
    C, T = data.shape
    w = window_sec * fs
    s = step_sec * fs
    starts = np.arange(0, T - w + 1, s, dtype=int)
    segments = np.stack([data[:, i:i + w] for i in starts], axis=0)  # (S, C, w)
    return segments

def nearest_idx(arr: np.ndarray, val: float) -> int:
    return int(np.argmin(np.abs(arr - val)))

def calculate_de(psd: np.ndarray) -> np.ndarray:
    """
    Differential Entropy: 0.5 * log(2πeσ^2)
    psd 形状: (..., F_band, T_frames)
    返回: (..., T_frames)
    """
    variance = np.var(psd, axis=-2, ddof=1) + 1e-5
    de = 0.5 * np.log(2 * np.pi * np.e * variance)
    return de

def base_homo_select(
    eeg_segments: np.ndarray,
    sample_rate: int,
    num_channel: int,
    order: int
):
    """
    输入:
        eeg_segments: (S, C, win_len)
    输出:
        base_freq: (S, C) 每段每通道基频（由频谱最大值求取后再对时间帧平均）
        f: (F,) 频率网格
        harm_freq: (S, C, order) 每段每通道的 2..(order+1) 次谐波频率(就近对齐到 f)
        zxx: (S, C, F, T_frames) 复谱
    """
    fs = sample_rate
    nperseg = int(round(CFG.stft_win_sec * fs))
    noverlap = int(round(CFG.stft_overlap_ratio * nperseg))
    nfft = CFG.stft_nfft or nperseg

    # scipy.signal.stft 会沿最后一个轴做 STFT；N-D 输入会广播其它轴
    f, t, zxx = stft(
        eeg_segments, fs=fs, window='hann',
        nperseg=nperseg, noverlap=noverlap, nfft=nfft, axis=-1, padded=False, boundary=None
    )
    # 现在 zxx 形状：(S, C, F, T_frames)
    power = np.abs(zxx) ** 2
    # 频率轴为 -2，时间帧为 -1
    base_freq_idx = power.argmax(axis=-2)     # (S, C, T_frames) → 每帧的最强频点
    # 对时间帧取平均，得到每段每通道一个基频
    base_freq = f.take(base_freq_idx, mode='clip').mean(axis=-1)  # (S, C)

    # 谐波频率（对齐到 f 网格最近邻）
    harm_freq = np.zeros((eeg_segments.shape[0], num_channel, order), dtype=float)
    for s in range(eeg_segments.shape[0]):
        for ch in range(num_channel):
            bf = base_freq[s, ch]
            for k in range(order):  # 2..(order+1) 次
                harmonic_f = (k + 2) * bf
                harm_freq[s, ch, k] = f[nearest_idx(f, harmonic_f)]
    return base_freq, f, harm_freq, zxx

def feature_extract(
    base_freq: np.ndarray, f: np.ndarray, harm_freq: np.ndarray, zxx: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    输入:
        base_freq: (S, C)
        f: (F,)
        harm_freq: (S, C, order)
        zxx: (S, C, F, T_frames)
    输出:
        base_de_features: (S, C, T_frames)
        harmon_de_features: (S, C, T_frames)
    """
    power = np.abs(zxx) ** 2
    S, C, F, T_frames = power.shape
    alpha = 1e-5

    # 对各通道统计区间（稳健一些）
    harm_mean = harm_freq.mean(axis=-1)  # (S, C) → 对谐波序求平均
    base_de_list: List[np.ndarray] = []
    harm_de_list: List[np.ndarray] = []

    f_min, f_max = float(f.min()), float(f.max())

    for ch in range(C):
        mu_b, sd_b = float(base_freq[:, ch].mean()), float(base_freq[:, ch].std())
        mu_h, sd_h = float(harm_mean[:, ch].mean()), float(harm_mean[:, ch].std())

        # base 频带
        base_low = max(0.5, mu_b - sd_b)
        base_high = min(f_max, max(base_low + CFG.min_band_width_hz, mu_b + sd_b))

        # harmonic 频带（不与 base 重叠）
        harm_low = max(base_high, mu_h - sd_h)
        harm_high = min(f_max, max(harm_low + CFG.min_band_width_hz, mu_h + sd_h))

        # 最近邻索引，并确保 i1 < i2
        i1, i2 = sorted([nearest_idx(f, base_low), nearest_idx(f, base_high)])
        if i1 == i2:
            i2 = min(i1 + 1, len(f) - 1)
            i1 = max(0, i2 - 1)

        j1, j2 = sorted([nearest_idx(f, harm_low), nearest_idx(f, harm_high)])
        if j1 == j2:
            j2 = min(j1 + 1, len(f) - 1)
            j1 = max(0, j2 - 1)

        psd_b = power[:, ch, i1:i2 + 1, :] + alpha  # (S, F_b, T)
        psd_h = power[:, ch, j1:j2 + 1, :] + alpha

        de_b = calculate_de(psd_b)  # (S, T)
        de_h = calculate_de(psd_h)  # (S, T)

        base_de_list.append(de_b)   # 通道维累积
        harm_de_list.append(de_h)

        print(f"[Ch {ch:02d}] Base [{f[i1]:.2f},{f[i2]:.2f}] Hz | Harm [{f[j1]:.2f},{f[j2]:.2f}] Hz")

    # 组装回 (S, C, T)
    base_de_features = np.transpose(np.array(base_de_list), (1, 0, 2))
    harmon_de_features = np.transpose(np.array(harm_de_list), (1, 0, 2))

    # 形状断言
    assert base_de_features.shape == (S, C, T_frames)
    assert harmon_de_features.shape == (S, C, T_frames)

    print(f"[Feature] Base/Harm DE shape: {base_de_features.shape}")
    return base_de_features, harmon_de_features

def phase_sync(de_feat: np.ndarray) -> np.ndarray:
    """
    标准 PLV 相位同步
    输入:
        de_feat: (C, T_feat)
    输出:
        M: (C, C) ∈ [0,1]
    """
    C, T = de_feat.shape
    M = np.zeros((C, C), dtype=float)
    # PLV: abs(mean(exp(1j * phase_diff)))
    for i in range(C):
        M[i, i] = 1.0
        for j in range(i + 1, C):
            phase_diff = np.angle(np.exp(1j * (de_feat[i] - de_feat[j])))
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            M[i, j] = M[j, i] = plv
    return M

def phase_graph(base_features: torch.Tensor, harm_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    输入:
        base_features: (subjects, segments, channels, T_feat) torch
        harm_features: 同上
    输出:
        base_graph, harm_graph: (subjects, segments, channels, channels) torch
    """
    base = base_features.detach().cpu().numpy()
    harm = harm_features.detach().cpu().numpy()
    Sbj, Seg, C, T = base.shape
    base_graph = np.empty((Sbj, Seg, C, C), dtype=float)
    harm_graph = np.empty_like(base_graph)

    for s in range(Sbj):
        for k in range(Seg):
            base_graph[s, k] = phase_sync(base[s, k])  # (C, T) → (C, C)
            harm_graph[s, k] = phase_sync(harm[s, k])

    return torch.from_numpy(base_graph), torch.from_numpy(harm_graph)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# =========================
# 主流程（逐被试处理并汇总）
# =========================
def process_one_subject(trial: dict, cfg: Config) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    输入 trial: {'data': np.ndarray, 'labels': np.ndarray}
    返回:
        base_de_features: (segments, channels, T_feat)
        harmon_de_features: (segments, channels, T_feat)
        y: int (0/1)
    """
    data = trial['data']  # 期望: (channels, time)
    labels = trial['labels']  # 期望: (40, 4) 或 (4,)
    fs = cfg.sample_rate

    # 1) 基线标定
    data = data_calibrate(data, fs, cfg.baseline_sec)  # → (channels, total_len)

    # 2) 分窗
    segments = data_divide(data, fs, cfg.window_sec, cfg.step_sec)  # (S, C, w)
    num_segments, num_channels, win_len = segments.shape
    assert num_channels == data.shape[0], "通道数不一致"
    print(f"[Divide] segments={num_segments}, channels={num_channels}, win_len={win_len}")

    # 3) STFT & 基频/谐波
    base_freq, f, harm_freq, zxx = base_homo_select(
        segments, sample_rate=fs, num_channel=num_channels, order=cfg.harmonic_order
    )

    # 4) 提取 DE 特征
    base_de, harm_de = feature_extract(base_freq, f, harm_freq, zxx)  # (S, C, T_feat)

    # 5) 标签（对当前 trial 的所有分段共用同一标签）
    y = set_label(np.asarray(labels), dim=cfg.label_dim, threshold=cfg.label_threshold)  # int 0/1

    return base_de, harm_de, y

def main():
    cfg = CFG
    ensure_dir(cfg.out_dir)

    file_names = sorted(os.listdir(cfg.raw_dir))
    all_base_de_features: List[torch.Tensor] = []
    all_harmon_de_features: List[torch.Tensor] = []
    all_labels_per_subject: List[torch.Tensor] = []

    for filename in file_names:
        filepath = os.path.join(cfg.raw_dir, filename)
        if not os.path.isfile(filepath) or not filename.endswith(".dat"):
            # 兼容不同扩展名；若你的原始文件不是 .dat，请去掉此判断
            # 也可以直接处理所有文件名
            pass
        print(f"\n******* Processing {filename} *******")
        trial = read_eeg_signal_from_file(filepath)

        base_de, harm_de, y = process_one_subject(trial, cfg)
        # 堆叠到 torch： (segments, channels, T_feat)
        base_de_t = torch.from_numpy(base_de).float()
        harm_de_t = torch.from_numpy(harm_de).float()

        # 统一到 (1, segments, channels, T_feat) 以便最终 cat
        all_base_de_features.append(base_de_t.unsqueeze(0))
        all_harmon_de_features.append(harm_de_t.unsqueeze(0))

        # per-segment labels（每个 segment 一个相同的标签），形状 (segments,)
        seg = base_de.shape[0]
        labels_t = torch.full((seg,), int(y), dtype=torch.long)
        all_labels_per_subject.append(labels_t.unsqueeze(0))  # (1, segments)

    # 汇总：subjects 维度
    all_base_de_features = torch.cat(all_base_de_features, dim=0)    # (subjects, segments, channels, T_feat)
    all_harmon_de_features = torch.cat(all_harmon_de_features, dim=0)
    all_labels = torch.cat(all_labels_per_subject, dim=0)            # (subjects, segments)

    print("\n=== Final Shapes ===")
    print("Base DE: ", tuple(all_base_de_features.shape))
    print("Harm DE: ", tuple(all_harmon_de_features.shape))
    print("Labels : ", tuple(all_labels.shape))

    # 保存特征与标签
    ensure_dir(cfg.out_dir)
    torch.save(all_base_de_features, os.path.join(cfg.out_dir, "all_base_de_features.pt"))
    torch.save(all_harmon_de_features, os.path.join(cfg.out_dir, "all_harmon_de_features.pt"))
    torch.save(all_labels, os.path.join(cfg.out_dir, "all_labels.pt"))

    # 计算图（PLV 相位同步）
    base_graph, harm_graph = phase_graph(all_base_de_features, all_harmon_de_features)
    torch.save(base_graph, os.path.join(cfg.out_dir, "base_graph.pt"))
    torch.save(harm_graph, os.path.join(cfg.out_dir, "harm_graph.pt"))

    print(f"\nSaved to {cfg.out_dir}")

if __name__ == "__main__":
    main()
