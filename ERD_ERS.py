import mne
import numpy as np
import pandas as pd
from scipy.signal import welch
from pathlib import Path
from imblearn.over_sampling import SMOTE
import json
from sklearn.preprocessing import StandardScaler

# ========== 基本参数 ==========
base_path = Path("E:/lin/stroke/stroke_GCN_ver3/data/")
mild_path = base_path / "mild"
severe_path = base_path / "severe"

channels = ['C3', 'C4']
freq_band = (31, 40)
trial_duration = 8
baseline_window = (6, 8)
active_window = (2, 6)
max_trials = 20

# ========== 计算 ERD/ERS% ==========
def calculate_erd_percentage(data, sfreq, baseline_idx, active_idx, freq_band):
    baseline_data = data[baseline_idx[0]:baseline_idx[1]]
    active_data = data[active_idx[0]:active_idx[1]]

    if len(baseline_data) < 50 or len(active_data) < 50:
        return np.nan

    f, P_baseline = welch(baseline_data, fs=sfreq, nperseg=min(128, len(baseline_data)))
    f, P_active = welch(active_data, fs=sfreq, nperseg=min(128, len(active_data)))

    band_mask = (f >= freq_band[0]) & (f <= freq_band[1])
    if np.sum(band_mask) == 0:
        return np.nan

    P_baseline_band = np.mean(P_baseline[band_mask])
    P_active_band = np.mean(P_active[band_mask])

    if P_baseline_band > 0:
        return (P_active_band - P_baseline_band) / P_baseline_band * 100
    else:
        return np.nan

# ========== 处理单个被试 ==========
def process_subject_file(file_path, group_name):
    results = []
    print(f"Processing {file_path.name}...")

    raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
    sfreq = raw.info['sfreq']
    samples_per_trial = int(trial_duration * sfreq)
    total_trials = len(raw.times) // samples_per_trial
    total_trials = min(total_trials, max_trials)

    picks = mne.pick_channels(raw.info['ch_names'], include=channels)
    if len(picks) < 2:
        print(f"  No C3/C4 channels found in {file_path}")
        return []

    for trial_idx in range(total_trials):
        trial_start = trial_idx * samples_per_trial
        trial_end = trial_start + samples_per_trial

        baseline_start = trial_start + int(baseline_window[0] * sfreq)
        baseline_end   = trial_start + int(baseline_window[1] * sfreq)
        active_start   = trial_start + int(active_window[0] * sfreq)
        active_end     = trial_start + int(active_window[1] * sfreq)

        trial_data, _ = raw[picks, trial_start:trial_end]

        erd_dict = {}
        for i, pick in enumerate(picks):
            ch_name = raw.info['ch_names'][pick]
            ch_data = trial_data[i]

            erd_value = calculate_erd_percentage(
                ch_data, sfreq,
                (baseline_start - trial_start, baseline_end - trial_start),
                (active_start - trial_start, active_end - trial_start),
                freq_band
            )
            erd_dict[ch_name] = erd_value

        if 'C3' in erd_dict and 'C4' in erd_dict:
            # 判断左右手
            hand_type = 'left' if (trial_idx+1) % 2 == 1 else 'right'
            if hand_type == 'left':
                erd_contra, erd_ipsi = erd_dict['C4'], erd_dict['C3']
            else:
                erd_contra, erd_ipsi = erd_dict['C3'], erd_dict['C4']

            denom = abs(erd_contra) + abs(erd_ipsi)
            LI = (erd_contra - erd_ipsi) / denom if denom > 0 else np.nan

            results.append({
                'subject': file_path.stem,
                'trial': trial_idx+1,
                'group': group_name,
                'hand': hand_type,
                'ERD_C3': erd_dict['C3'],
                'ERD_C4': erd_dict['C4'],
                'LI': LI
            })

    return results

# ========== 主流程 ==========
all_results = []
for file_path in mild_path.glob("*.fif"):
    all_results.extend(process_subject_file(file_path, "mild"))
for file_path in severe_path.glob("*.fif"):
    all_results.extend(process_subject_file(file_path, "severe"))

df = pd.DataFrame(all_results)
print(f"处理完成: {len(df)} 条记录, {len(df['subject'].unique())} 个受试者")

# ========== SMOTE 扩增 ==========
X = df[['LI']].values
y = df['group'].map({'mild': 0, 'severe': 1}).values

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

df_res = pd.DataFrame({
    'LI': X_res.flatten(),
    'group': ['mild' if i == 0 else 'severe' for i in y_res]
})
print(f"扩增完成: {df_res.shape[0]} 条记录")

# ========== 保存被试字典 ==========
li_dict = {}
for subj in df['subject'].unique():
    li_dict[subj] = df[df['subject'] == subj]['LI'].dropna().tolist()

with open("erd_ers_results.txt", "w", encoding="utf-8") as f:
    json.dump(li_dict, f, indent=4, ensure_ascii=False)
print("每个被试的LI字典已保存 -> erd_ers_results.txt")

# ========== 归一化 & 保存npy ==========
scaler = StandardScaler()
LI_features = scaler.fit_transform(df_res[['LI']].values)

np.save("LI_features.npy", LI_features)
print(f"归一化后的 LI 特征已保存 -> LI_features.npy, 形状: {LI_features.shape}")