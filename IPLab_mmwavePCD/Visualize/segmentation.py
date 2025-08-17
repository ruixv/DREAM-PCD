import numpy as np
import torch

def save_point_cloud_with_labels(data_save, seg, pred):

    assert data_save.size() == (16, 4096, 9)
    assert seg.size() == (16, 4096)
    assert pred.size() == (16, 4096)
    
    for idx in range(data_save.size(0)):
        sample = data_save[idx].cpu().numpy()
        gt_label = seg[idx].cpu().numpy().reshape(-1, 1) 
        pred_label = pred[idx].cpu().numpy().reshape(-1, 1)  

        sample_with_labels = np.hstack((sample, gt_label, pred_label))
        
        sample_with_labels.tofile(f'frame_{idx}.bin')

# save_point_cloud_with_labels(data_save, seg, pred)