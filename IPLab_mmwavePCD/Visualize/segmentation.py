import numpy as np
import torch

def save_point_cloud_with_labels(data_save, seg, pred):
    # 确保输入的形状匹配
    assert data_save.size() == (16, 4096, 9)
    assert seg.size() == (16, 4096)
    assert pred.size() == (16, 4096)
    
    for idx in range(data_save.size(0)):
        sample = data_save[idx].cpu().numpy()  # 将点云转换为numpy格式
        gt_label = seg[idx].cpu().numpy().reshape(-1, 1)  # 调整GT标签的形状
        pred_label = pred[idx].cpu().numpy().reshape(-1, 1)  # 调整预测标签的形状

        # 将 GT 标签和预测标签添加到点云数据中
        sample_with_labels = np.hstack((sample, gt_label, pred_label))
        
        # 保存点云数据为 .bin 文件
        sample_with_labels.tofile(f'frame_{idx}.bin')

# 调用函数
# save_point_cloud_with_labels(data_save, seg, pred)