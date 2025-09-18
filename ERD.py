



import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import welch
import os
from pathlib import Path
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# 如果需要嵌入字体，可以设置使用系统可识别的字体
mpl.rcParams['font.family'] = 'Times New Roman'  # 或其他常用字体
# 设置样式
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

# 1. 定义文件路径
base_path = Path("E:/lin/stroke/stroke_GCN_ver3/data/")
mild_path = base_path / "mild"
severe_path = base_path / "severe"

# 2. 定义参数
freq_bands = {'alpha': (8, 12), 'beta': (13, 30)}
channels = ['C3', 'C4']
trial_duration = 8  # 每个试次8秒
instruction_duration = 2  # 指令2秒
mi_duration = 4  # 运动想象4秒
rest_duration = 2  # 休息2秒

# 时间窗口定义（相对于每个试次的开始）
baseline_window = (6, 8)  # 休息期作为基线：6-8秒
active_window = (2, 6)  # 运动想象期：2-6秒


def calculate_erd_percentage(data, sfreq, baseline_idx, active_idx, freq_band):
    """
    计算ERD%百分比
    """
    # 计算基线期功率
    baseline_data = data[baseline_idx[0]:baseline_idx[1]]
    if len(baseline_data) < 50:  # 确保有足够的数据点
        return np.nan

    f, P_baseline = welch(baseline_data, fs=sfreq, nperseg=min(128, len(baseline_data)))

    # 计算活动期功率
    active_data = data[active_idx[0]:active_idx[1]]
    if len(active_data) < 50:
        return np.nan

    f, P_active = welch(active_data, fs=sfreq, nperseg=min(128, len(active_data)))

    # 提取目标频带功率
    band_mask = (f >= freq_band[0]) & (f <= freq_band[1])
    if np.sum(band_mask) == 0:
        return np.nan

    P_baseline_band = np.mean(P_baseline[band_mask])
    P_active_band = np.mean(P_active[band_mask])

    # 计算ERD%
    if P_baseline_band > 0:
        erd_percentage = ((P_active_band - P_baseline_band) / P_baseline_band) * 100
    else:
        erd_percentage = np.nan

    return erd_percentage


def is_contralateral(channel, hand_type):
    """判断是否为对侧通道"""
    if hand_type == 'left':
        return channel == 'C4'  # 左手想象，对侧是C4（右脑）
    else:
        return channel == 'C3'  # 右手想象，对侧是C3（左脑）


def process_subject_file(file_path, group_name):
    """
    处理单个被试文件的20个试次
    """
    try:
        print(f"Processing {file_path.name}...")

        # 读取数据
        raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
        sfreq = raw.info['sfreq']

        # 计算每个试次的样本数
        samples_per_trial = int(trial_duration * sfreq)
        total_trials = len(raw.times) // samples_per_trial
        total_trials = min(total_trials, 20)  # 最多处理20个试次

        print(f"  找到 {total_trials} 个试次，采样率 {sfreq} Hz")

        # 提取C3、C4通道数据
        picks = mne.pick_channels(raw.info['ch_names'], include=channels)
        if len(picks) == 0:
            print(f"  No C3/C4 channels found in {file_path}")
            return []

        results = []

        for trial_idx in range(total_trials):
            # 确定试次开始和结束位置
            trial_start = trial_idx * samples_per_trial
            trial_end = trial_start + samples_per_trial

            # 确定试次类型：奇数试次左手，偶数试次右手
            hand_type = 'left' if (trial_idx % 2) == 0 else 'right'

            # 计算时间窗口索引（相对于试次开始）
            baseline_start = trial_start + int(baseline_window[0] * sfreq)
            baseline_end = trial_start + int(baseline_window[1] * sfreq)
            active_start = trial_start + int(active_window[0] * sfreq)
            active_end = trial_start + int(active_window[1] * sfreq)

            # 确保索引不越界
            baseline_start = max(trial_start, baseline_start)
            baseline_end = min(trial_end, baseline_end)
            active_start = max(trial_start, active_start)
            active_end = min(trial_end, active_end)

            # 提取该试次的数据
            trial_data, _ = raw[picks, trial_start:trial_end]

            for i, pick in enumerate(picks):
                ch_name = raw.info['ch_names'][pick]
                ch_data = trial_data[i]

                for band_name, freq_range in freq_bands.items():
                    # 计算ERD%
                    erd_value = calculate_erd_percentage(
                        ch_data, sfreq,
                        (baseline_start - trial_start, baseline_end - trial_start),
                        (active_start - trial_start, active_end - trial_start),
                        freq_range
                    )

                    if not np.isnan(erd_value):
                        contralateral = is_contralateral(ch_name, hand_type)

                        results.append({
                            'subject': file_path.stem,
                            'trial': trial_idx + 1,
                            'hand_type': hand_type,
                            'channel': ch_name,
                            'frequency_band': band_name,
                            'erd_percentage': erd_value,
                            'contralateral': contralateral,
                            'group': group_name,
                            'file_path': str(file_path)
                        })

        print(f"  成功处理 {len(results)} 条记录")
        return results

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return []


# 首先检查文件路径
print("Check the file path...")
print(f"Base path: {base_path}")
print(f"Mild path exists: {mild_path.exists()}")
print(f"Severe path exists: {severe_path.exists()}")

if not mild_path.exists() or not severe_path.exists():
    print("Please check if the file path is correct！")
    exit()

# 查找.fif文件
mild_files = list(mild_path.glob("*.fif"))
severe_files = list(severe_path.glob("*.fif"))

print(f"find {len(mild_files)} mild file")
print(f"find {len(severe_files)} moderate file")

if len(mild_files) == 0 and len(severe_files) == 0:
    print("The.fif file was not found")
    exit()

# 3. 主处理流程
all_results = []

# 处理mild组
print("\nProcessing mild group...")
for file_path in mild_files:
    results = process_subject_file(file_path, 'mild')
    all_results.extend(results)

# 处理severe组
print("\nProcessing severe group...")
for file_path in severe_files:
    results = process_subject_file(file_path, 'severe')
    all_results.extend(results)

# 检查是否处理了数据
if not all_results:
    print("No data processed.")
else:
    # 转换为DataFrame
    df = pd.DataFrame(all_results)
    print(f"\nSuccessfully processed {len(df)} trial records from {len(set(df['subject']))} subjects")

    # 显示数据概览
    print("\nData Overview:")
    print(f"Total number of trials: {len(df)}")
    print(f"The number of trials in the mild group: {len(df[df['group'] == 'mild'])}")
    print(f"The number of trials in the moderate group: {len(df[df['group'] == 'moderate'])}")
    print(f"The number of trials with the left hand: {len(df[df['hand_type'] == 'left'])}")
    print(f"The number of trials with the right hand: {len(df[df['hand_type'] == 'right'])}")

    # 计算每个被试的平均ERD%
    subject_avg_df = df.groupby(['subject', 'group', 'hand_type', 'channel', 'frequency_band', 'contralateral'])[
        'erd_percentage'].mean().reset_index()

    print(f"\nThe average number of records for each subject: {len(subject_avg_df)}")
    print(f"The average number of records in the mild group: {len(subject_avg_df[subject_avg_df['group'] == 'mild'])}")
    print(f"The average number of records in the moderate group: {len(subject_avg_df[subject_avg_df['group'] == 'moderate'])}")

    # 4. 统计分析和可视化
    if 'contralateral' in subject_avg_df.columns:
        contralateral_df = subject_avg_df[subject_avg_df['contralateral'] == True]
        print(f"\nThe number of records in the opposite channel: {len(contralateral_df)}")

        if len(contralateral_df) > 0:
            print("\n=== Statistical results (based on the average value of the subjects) ===")

            # 按组和频段显示描述性统计
            stats_summary = contralateral_df.groupby(['group', 'frequency_band'])['erd_percentage'].agg(
                ['mean', 'std', 'count'])
            print("\nDescriptive statistics of the contralateral ERD%:")
            print(stats_summary)

            # 统计检验
            print("\nInter-group comparison (independent sample t-test):")
            for band in freq_bands.keys():
                mild_data = contralateral_df[(contralateral_df['group'] == 'mild') &
                                             (contralateral_df['frequency_band'] == band)]['erd_percentage']
                severe_data = contralateral_df[(contralateral_df['group'] == 'severe') &
                                               (contralateral_df['frequency_band'] == band)]['erd_percentage']

                if len(mild_data) > 1 and len(severe_data) > 1:
                    t_stat, p_value = stats.ttest_ind(mild_data, severe_data)
                    print(f"{band}-band: t({len(mild_data) + len(severe_data) - 2}) = {t_stat:.3f}, p = {p_value:.4f}")

                    # 计算效应量 (Cohen's d)
                    pooled_std = np.sqrt((mild_data.std() ** 2 + severe_data.std() ** 2) / 2)
                    cohens_d = (mild_data.mean() - severe_data.mean()) / pooled_std
                    print(f"  Cohen's d = {cohens_d:.3f}")

                    # 解释结果
                    if p_value < 0.05:
                        if mild_data.mean() < severe_data.mean():
                            print(f"  The ERD effect in the mild group was significantly stronger than that in the moderate group (with a more negative ERD%).")
                        else:
                            print(f"  The ERD effect in the moderate group was significantly stronger than that in the mild group")
                    else:
                        print(f"  There was no significant difference between groups")

            # 5. 可视化
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('ERD analysis results of motor imagery (based on the average value of the subjects)', fontsize=16)

            # 5.1 对侧ERD%分布箱线图
            plt.sca(axes[0, 0])
            sns.boxplot(data=contralateral_df, x='frequency band', y='erd percentage', hue='group')
            plt.title('Contralateral ERD% distribution (by group and frequency band)')
            plt.ylabel('ERD%')
            plt.axhline(0, color='red', linestyle='--', alpha=0.5, label='Baseline level')
            plt.legend()

            # 5.2 左右手ERD%比较
            plt.sca(axes[0, 1])
            hand_comparison = contralateral_df.groupby(['group', 'hand type'])['erd percentage'].mean().unstack()
            if not hand_comparison.empty:
                hand_comparison.plot(kind='bar', ax=axes[0, 1])
                plt.title('Comparison of Contralateral ERD% of the left and right hands')
                plt.ylabel('Mean ERD%')
                plt.xlabel('Group')
                plt.xticks(rotation=0)
                plt.axhline(0, color='red', linestyle='--', alpha=0.5)

            # 5.3 各组平均ERD%比较
            plt.sca(axes[1, 0])
            group_band_means = contralateral_df.groupby(['group', 'frequency band'])['erd percentage'].mean().unstack()
            if not group_band_means.empty:
                group_band_means.plot(kind='bar', ax=axes[1, 0])
                plt.title('Comparison of average contralateral ERD%')
                plt.ylabel('Mean ERD%')
                plt.xlabel('Group')
                plt.xticks(rotation=0)
                plt.axhline(0, color='red', linestyle='--', alpha=0.5)

            # 5.4 个体差异展示
            plt.sca(axes[1, 1])
            # 选择alpha频段展示个体差异
            alpha_data = contralateral_df[contralateral_df['frequency band'] == 'alpha']
            subjects = sorted(set(alpha_data['subject']))
            mild_values = []
            severe_values = []

            for subject in subjects:
                sub_data = alpha_data[alpha_data['subject'] == subject]
                if len(sub_data) > 0:
                    if sub_data['group'].iloc[0] == 'mild':
                        mild_values.append(sub_data['erd percentage'].mean())
                    else:
                        severe_values.append(sub_data['erd percentage'].mean())

            plt.scatter(['Mild'] * len(mild_values), mild_values, alpha=0.6, s=50)
            plt.scatter(['Moderate'] * len(severe_values), severe_values, alpha=0.6, s=50)
            plt.title('Individual ERD% value distribution (Alpha band)')
            plt.ylabel('ERD%')
            plt.axhline(0, color='red', linestyle='--', alpha=0.5)

            plt.tight_layout()
            plt.savefig('ERD_analysis_results.png', dpi=300, bbox_inches='tight')
            plt.savefig("ERD_analysis_results.pdf", format='pdf', bbox_inches='tight')
            plt.show()

            # 6. 保存结果到CSV
            df.to_csv('erd_analysis_all_trials.csv', index=False)
            subject_avg_df.to_csv('erd_analysis_subject_avg.csv', index=False)
            contralateral_df.to_csv('contralateral_erd_results.csv', index=False)

            print("\nAnalysis completed! The result has been saved to the CSV file")
            print("  - erd_analysis_all_trials.csv: The original data of all the trials")
            print("  - erd_analysis_subject_avg.csv: The average value of each subject")
            print("  - contralateral_erd_results.csv: Analysis results of the contralateral channel")

            # 显示主要发现摘要
            print("\n=== Summary of Main Findings ===")
            for band in freq_bands.keys():
                mild_mean = contralateral_df[(contralateral_df['group'] == 'mild') &
                                             (contralateral_df['frequency_band'] == band)]['erd_percentage'].mean()
                severe_mean = contralateral_df[(contralateral_df['group'] == 'severe') &
                                               (contralateral_df['frequency_band'] == band)]['erd_percentage'].mean()
                print(f"{band} band: The average ERD% of the mild group = {mild_mean:.2f}%, The average ERD% of the moderate group = {severe_mean:.2f}%")

                # ERD%值越负表示ERD效应越强
                if mild_mean < severe_mean:
                    print(f" The ERD effect in the mild group was stronger than that in the moderate group {severe_mean - mild_mean:.2f}%")
                else:
                    print(f" The ERD effect in the moderate group was stronger than that in the mild group {mild_mean - severe_mean:.2f}%")

    else:
        print("The 'contralateral' column was not found in the data frame.")
        print("Available columns:", subject_avg_df.columns.tolist())