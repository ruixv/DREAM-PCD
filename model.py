#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM

Modified by 
@Author: An Tao, Ziyi Wu
@Contact: ta19@mails.tsinghua.edu.cn, dazitu616@gmail.com
@Time: 2022/7/30 7:49 PM

Modified by 
@Author: Ruixu Geng
@Contact: gengruixu@mail.ustc.edu.cn
@Time: 2023/06/04 22:37 PM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

def knn(x, k):
    """
    Compute k nearest neighbors for each point in the point cloud.
    Args:
        x: input points, [B, C, N]
        k: number of neighbors
    Returns:
        idx: indices of k-nearest neighbors, [B, N, k]
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # [B, N, k]
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    """
    Construct edge features for each point.
    Args:
        x: input points, [B, C, N]
        k: number of neighbors
        idx: indices of neighbors, if None, compute using knn
        dim9: whether to use the last 3 dimensions for neighbor finding
    Returns:
        edge features: [B, 2C, N, k]
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    # pdb.set_trace()
    if idx is None:
        if not dim9:
            idx = knn(x, k=k)   # [B, N, k]
        else:
            idx = knn(x[:, :3, :], k=k)  # Use only the last 3 dimensions
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # [B, N, C]
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature  # [B, 2*C, N, k]


class DGCNN_Base(nn.Module):
    """
    Base DGCNN model for semantic segmentation
    """
    def __init__(self, args):
        """
        Initialize DGCNN semantic segmentation model.
        Args:
            args: arguments containing:
                k: number of nearest neighbors
                num_features: number of input features
                num_labels: number of output classes
                emb_dims: dimensions of embedding
                dropout: dropout rate
        """
        super(DGCNN_Base, self).__init__()
        self.args = args
        self.k = args.k
        self.num_features = args.num_features
        self.num_labels = args.num_labels
        
        # BatchNorm layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        # Convolutional layers
        self.conv1 = nn.Sequential(nn.Conv2d(2*self.num_features, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        # Dropout and final prediction layer
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Conv1d(256, self.num_labels, kernel_size=1, bias=False)
        
    def forward(self, x):
        """
        Forward pass of the network.
        Args:
            x: input point cloud, [B, C, N] where C is the number of features and N is the number of points
        Returns:
            features: feature representation before final prediction, [B, 256, N]
            logits: prediction logits, [B, num_labels, N]
        """
        batch_size = x.size(0)
        num_points = x.size(2)

        # First EdgeConv block
        
        x = get_graph_feature(x, k=self.k, dim9=False)   # [B, 2C, N, k]
        x = self.conv1(x)                               # [B, 64, N, k]
        x = self.conv2(x)                               # [B, 64, N, k]
        x1 = x.max(dim=-1, keepdim=False)[0]            # [B, 64, N]

        # Second EdgeConv block
        x = get_graph_feature(x1, k=self.k)             # [B, 128, N, k]
        x = self.conv3(x)                               # [B, 64, N, k]
        x = self.conv4(x)                               # [B, 64, N, k]
        x2 = x.max(dim=-1, keepdim=False)[0]            # [B, 64, N]

        # Third EdgeConv block
        x = get_graph_feature(x2, k=self.k)             # [B, 128, N, k]
        x = self.conv5(x)                               # [B, 64, N, k]
        x3 = x.max(dim=-1, keepdim=False)[0]            # [B, 64, N]

        # Concatenate features from different layers
        x = torch.cat((x1, x2, x3), dim=1)              # [B, 192, N]

        # Global feature extraction
        x = self.conv6(x)                               # [B, emb_dims, N]
        x = x.max(dim=-1, keepdim=True)[0]              # [B, emb_dims, 1]

        # Feature propagation
        x = x.repeat(1, 1, num_points)                  # [B, emb_dims, N]
        x = torch.cat((x, x1, x2, x3), dim=1)           # [B, emb_dims+192, N]

        # Final MLP
        x = self.conv7(x)                               # [B, 512, N]
        x = self.conv8(x)                               # [B, 256, N]
        features = x                                    # Store features before final layer
        
        x = self.dp1(x)
        logits = self.conv9(x)                          # [B, num_labels, N]
        
        return features, logits


class KalmanFusionModule(nn.Module):
    """
    Kalman Filter based fusion module for point cloud features
    """
    def __init__(self, feature_dim):
        super(KalmanFusionModule, self).__init__()
        self.feature_dim = feature_dim
        
        # Learnable parameters for Kalman filter
        self.Q = nn.Parameter(torch.ones(1, feature_dim, 1) * 0.01)  # Process noise
        self.R1 = nn.Parameter(torch.ones(1, feature_dim, 1) * 0.1)  # Measurement noise for input 1
        self.R2 = nn.Parameter(torch.ones(1, feature_dim, 1) * 0.1)  # Measurement noise for input 2
        
        # Adaptation layer for feature calibration
        self.adaptation = nn.Sequential(
            nn.Conv1d(feature_dim * 2, feature_dim, kernel_size=1),
            nn.BatchNorm1d(feature_dim),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
    def forward(self, x1, x2):
        """
        Fuse two feature maps using Kalman filtering
        Args:
            x1: First feature map [B, C, N]
            x2: Second feature map [B, C, N]
        Returns:
            fused: Fused feature map [B, C, N]
        """
        batch_size, _, num_points = x1.size()
        
        # Initial state estimate (x_hat) is the adapted combination of both inputs
        combined = torch.cat((x1, x2), dim=1)
        x_hat = self.adaptation(combined)
        
        # Initial error covariance (P)
        P = torch.ones_like(x_hat) * 1.0
        
        # Kalman gain for first measurement
        K1 = P / (P + self.R1)
        
        # Update state estimate with first measurement
        x_hat = x_hat + K1 * (x1 - x_hat)
        
        # Update error covariance
        P = (1 - K1) * P
        
        # Add process noise
        P = P + self.Q
        
        # Kalman gain for second measurement
        K2 = P / (P + self.R2)
        
        # Update state estimate with second measurement
        x_hat = x_hat + K2 * (x2 - x_hat)
        
        # Update error covariance
        P = (1 - K2) * P
        
        return x_hat


class RDM_Network(nn.Module):
    """
    RDM Network with dual inputs and Kalman fusion
    """
    def __init__(self, args):
        super(RDM_Network, self).__init__()
        self.args = args
        
        # Two identical DGCNN networks for the two input point clouds
        self.dgcnn1 = DGCNN_Base(args)
        self.dgcnn2 = DGCNN_Base(args)
        
        # Kalman fusion module
        self.fusion = KalmanFusionModule(feature_dim=256)
        
        # Final classification layer (after fusion)
        self.final_layer = nn.Conv1d(256, args.num_labels, kernel_size=1, bias=False)
        self.dp = nn.Dropout(p=args.dropout)
        
    def forward(self, x1, x2):
        """
        Forward pass of the RDM network.
        Args:
            x1: First input point cloud, [B, C, N]
            x2: Second input point cloud, [B, C, N]
        Returns:
            logits: prediction logits after fusion, [B, num_labels, N]
        """
        # Process first point cloud
        features1, logits1 = self.dgcnn1(x1)
        
        # Process second point cloud
        features2, logits2 = self.dgcnn2(x2)
        
        # Fuse features using Kalman filter
        fused_features = self.fusion(features1, features2)
        
        # Final prediction layer
        fused_features = self.dp(fused_features)
        fused_logits = self.final_layer(fused_features)
        
        return fused_logits


class DGCNN_semseg_s3dis(nn.Module):
    """
    DGCNN model for semantic segmentation on S3DIS dataset.
    Kept for compatibility with original code.
    """
    def __init__(self, args):
        super(DGCNN_semseg_s3dis, self).__init__()
        self.args = args
        self.base_model = DGCNN_Base(args)
        
    def forward(self, x):
        """
        Forward pass of the network.
        Args:
            x: input point cloud, [B, C, N]
        Returns:
            logits: prediction logits, [B, num_labels, N]
        """
        _, logits = self.base_model(x)
        return logits