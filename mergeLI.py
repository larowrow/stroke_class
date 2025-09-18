import numpy as np
import os


def enhance_features_with_li(mi_data, li_data, c3_index=13, c4_index=14):
    """
    使用LI值增强特征矩阵

    参数:
    mi_data: 原始特征矩阵 (720, 348) = (720, 29*12)
    li_data: LI值矩阵 (720, 1)
    c3_index: C3节点的索引 (0-based)
    c4_index: C4节点的索引 (0-based)

    返回:
    enhanced_data: 增强后的特征矩阵 (720, 29*13)
    """
    # 获取样本数量
    num_samples = mi_data.shape[0]
    num_nodes = 29
    num_original_features = 12
    num_enhanced_features = 13

    # 重塑原始数据为 (720, 29, 12)
    original_reshaped = mi_data.reshape(num_samples, num_nodes, num_original_features)

    # 初始化增强后的特征矩阵 (720, 29, 13)
    enhanced_reshaped = np.zeros((num_samples, num_nodes, num_enhanced_features))

    # 确保li_data是正确形状 (720,)
    if li_data.ndim == 2:
        li_values = li_data.flatten()
    else:
        li_values = li_data

    for sample_idx in range(num_samples):
        # 复制原始特征到前12个维度
        enhanced_reshaped[sample_idx, :, :num_original_features] = original_reshaped[sample_idx]

        # 获取当前样本的LI值
        li_value = li_values[sample_idx]

        # C3节点：原始12特征 + LI值
        enhanced_reshaped[sample_idx, c3_index, num_original_features] = li_value

        # C4节点：原始12特征 + (-LI值)
        enhanced_reshaped[sample_idx, c4_index, num_original_features] = -li_value

        # 其他节点：原始12特征 + 0（保持为0）
        # 注意：其他节点的第13维已经是0，无需额外操作

    # 重塑回二维矩阵 (720, 29*13)
    enhanced_data = enhanced_reshaped.reshape(num_samples, num_nodes * num_enhanced_features)

    return enhanced_data


def main():
    # 文件路径
    mi_data_path = r"E:\lin\stroke\stroke_GCN_ver3\data\time_8\Gamma\MI_plv_net_time_freq.npy"
    li_data_path = r"E:\lin\stroke\stroke_GCN_ver3\data\time_8\Gamma_LI\LI_features.npy"
    output_path = r"E:\lin\stroke\stroke_GCN_ver3\data\time_8\Gamma_LI\MI_plv_net_time_freq_LI.npy"

    # 加载数据
    print("正在加载数据...")
    mi_data = np.load(mi_data_path)
    li_data = np.load(li_data_path)

    print(f"原始MI数据维度: {mi_data.shape}")
    print(f"LI数据维度: {li_data.shape}")

    # 节点索引（0-based）
    C3_index = 13  # C3通道对应第14个节点（索引13）
    C4_index = 14  # C4通道对应第15个节点（索引14）

    # 增强特征
    print("正在增强特征...")
    enhanced_data = enhance_features_with_li(mi_data, li_data, C3_index, C4_index)

    print(f"增强后数据维度: {enhanced_data.shape}")
    print(f"预期维度: (720, {29 * 13}) = (720, 377)")

    # 验证数据
    print("\n验证增强结果:")
    print(f"样本0 - C3节点LI值: {enhanced_data[0, C3_index * 13 + 12]}")
    print(f"样本0 - C4节点LI值: {enhanced_data[0, C4_index * 13 + 12]}")
    print(f"样本0 - 其他节点LI值: {enhanced_data[0, 0 * 13 + 12]}")  # 检查第一个节点的LI维度

    # 保存增强后的数据
    print(f"\n保存数据到: {output_path}")
    np.save(output_path, enhanced_data)

    # 验证保存的数据
    print("验证保存的数据...")
    loaded_data = np.load(output_path)
    print(f"加载的数据维度: {loaded_data.shape}")

    # 检查数据一致性
    if np.array_equal(enhanced_data, loaded_data):
        print("✓ 数据保存和加载验证成功！")
    else:
        print("✗ 数据保存验证失败！")

    print("处理完成！")


if __name__ == "__main__":
    main()