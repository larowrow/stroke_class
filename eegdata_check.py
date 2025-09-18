import mne

# 加载数据
input_path = "data\severe\sub-40_task-motor-imagery_eeg_processed.fif"
raw = mne.io.read_raw_fif(input_path, preload=True)

# 删除指定通道
channels_to_drop = ['HEOL', 'HEOR', '']
channels_to_drop = [ch for ch in channels_to_drop if ch in raw.ch_names]
raw.drop_channels(channels_to_drop)

# 保存新文件
output_path = "sub-40_task-motor-imagery_eeg.fif"
raw.save(output_path, overwrite=True)
print(f"已移除 {channels_to_drop} 并保存到 {output_path}")

import mne

# 加载 FIF 文件
file_path = "data/severe/sub-11_task-motor-imagery_eeg.fif"
raw = mne.io.read_raw_fif(file_path, preload=True)  # preload=True 将数据加载到内存

# 打印文件基本信息
print(raw)
print("\n通道名称:", raw.ch_names)
print("采样频率:", raw.info["sfreq"], "Hz")
print("数据时长:", raw.times[-1], "秒")