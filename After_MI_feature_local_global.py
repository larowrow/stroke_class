import os
import numpy as np

# 设置数据目录和子文件夹
base_dir = "data/time_8"
subfolders = ["Alpha", "Beta", "Delta", "Theta", "Gamma"]

# 遍历每个子文件夹
for folder in subfolders:
    folder_path = os.path.join(base_dir, folder)

    # 处理 pcc 数据
    pcc_path = os.path.join(folder_path, "MI_pcc_net_time_freq.npy")
    if os.path.exists(pcc_path):
        pcc_data = np.load(pcc_path)  # (720, 348)
        pcc_data = pcc_data.reshape(pcc_data.shape[0], 29, 12)  # (720, 29, 12)

        pcc_local = pcc_data[:, :, :4].reshape(pcc_data.shape[0], -1)   # (720, 116)
        pcc_global = pcc_data[:, :, -8:].reshape(pcc_data.shape[0], -1) # (720, 232)

        np.save(os.path.join(folder_path, "pcc_localfeature.npy"), pcc_local)
        np.save(os.path.join(folder_path, "pcc_globalfeature.npy"), pcc_global)
        print(f"[{folder}] Saved: pcc_localfeature.npy, pcc_globalfeature.npy")

    # 处理 plv 数据
    plv_path = os.path.join(folder_path, "MI_plv_net_time_freq.npy")
    if os.path.exists(plv_path):
        plv_data = np.load(plv_path)  # (720, 348)
        plv_data = plv_data.reshape(plv_data.shape[0], 29, 12)  # (720, 29, 12)

        plv_local = plv_data[:, :, :4].reshape(plv_data.shape[0], -1)   # (720, 116)
        plv_global = plv_data[:, :, -8:].reshape(plv_data.shape[0], -1) # (720, 232)

        np.save(os.path.join(folder_path, "plv_localfeature.npy"), plv_local)
        np.save(os.path.join(folder_path, "plv_globalfeature.npy"), plv_global)
        print(f"[{folder}] Saved: plv_localfeature.npy, plv_globalfeature.npy")
