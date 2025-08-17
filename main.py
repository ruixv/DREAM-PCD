#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Point Cloud Semantic Segmentation using DGCNN and RDM

Original Authors: An Tao, Pengliang Ji
Modified by: Ruixu Geng
Further Modified for RDM Network
"""

from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data import RadarEyes
from model import DGCNN_semseg_s3dis, RDM_Network
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
import pdb
# Set environment variables for CUDA
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def _init_():
    """Initialize directory structure for saving model and outputs"""
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/'+args.exp_name):
        os.makedirs('outputs/'+args.exp_name)
    if not os.path.exists('outputs/'+args.exp_name+'/'+'models'):
        os.makedirs('outputs/'+args.exp_name+'/'+'models')
    
    # Create backup files of the code
    os.system('cp main.py outputs'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model.py outputs' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py outputs' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py outputs' + '/' + args.exp_name + '/' + 'data.py.backup')


def calculate_sem_IoU(pred_np, seg_np):
    """
    Calculate semantic IoU (Intersection over Union) for segmentation
    
    Args:
        pred_np: Prediction numpy array
        seg_np: Ground truth segmentation numpy array
    
    Returns:
        IoU values for each class
    """
    I_all = np.zeros(2)
    U_all = np.zeros(2)
    for sem_idx in range(seg_np.shape[0]):
        for sem in range(2):
            I = np.sum(np.logical_and(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            U = np.sum(np.logical_or(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            I_all[sem] += I
            U_all[sem] += U
    return I_all / U_all 


class DualRadarEyes(RadarEyes):
    """
    Dataset class for handling dual point cloud inputs.
    Inherits from RadarEyes and adds functionality for second point cloud.
    """
    def __init__(self, partition, num_points, scenes_list, data_type, secondary_data_type=None):
        super(DualRadarEyes, self).__init__(partition, num_points, scenes_list, data_type)
        self.secondary_data_type = secondary_data_type or data_type
        
    def __getitem__(self, item):
        # Get primary point cloud
        primary_data, seg = super(DualRadarEyes, self).__getitem__(item)
        
        # Get secondary point cloud (use a different data type if specified)
        # This is a placeholder - you might need to adjust based on your actual data structure
        # For example, you might have a different folder for the second point cloud type
        secondary_data_path = self.data_list[item].replace(self.data_type, self.secondary_data_type)
        secondary_data = self._load_data(secondary_data_path)
        
        return primary_data, secondary_data, seg
    
    def _load_data(self, path):
        # Implement custom data loading for secondary point cloud if needed
        # This is a placeholder function
        # For simplicity, we'll assume it works the same way as the primary data loading
        return super()._load_data(path)


def train(args, io):
    """
    Main training function
    
    Args:
        args: Command line arguments
        io: IO stream for logging
    """
    # Initialize data loaders
    if args.dual_input:
        train_dataset = DualRadarEyes(
            partition='train', 
            num_points=args.num_points,  
            scenes_list=args.scenes_list, 
            data_type=args.data_type,
            secondary_data_type=args.secondary_data_type
        )
        test_dataset = DualRadarEyes(
            partition='test', 
            num_points=args.num_points, 
            scenes_list=args.scenes_list, 
            data_type=args.data_type,
            secondary_data_type=args.secondary_data_type
        )
    else:
        train_dataset = RadarEyes(
            partition='train', 
            num_points=args.num_points,  
            scenes_list=args.scenes_list, 
            data_type=args.data_type
        )
        test_dataset = RadarEyes(
            partition='test', 
            num_points=args.num_points, 
            scenes_list=args.scenes_list, 
            data_type=args.data_type
        )
    
    train_loader = DataLoader(
        train_dataset,
        num_workers=args.num_workers, 
        batch_size=args.batch_size, 
        shuffle=True, 
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        num_workers=args.num_workers, 
        batch_size=args.test_batch_size, 
        shuffle=False, 
        drop_last=False
    )
    
    # Set device
    device = torch.device("cuda" if args.cuda else "cpu")

    # Initialize model
    if args.model == 'dgcnn' and not args.dual_input:
        model = DGCNN_semseg_s3dis(args).to(device)
        io.cprint("Using DGCNN model with single input")
    elif args.model == 'dgcnn' and args.dual_input:
        model = RDM_Network(args).to(device)
        io.cprint("Using RDM Network with dual input")
    else:
        raise Exception("Model not implemented")
    
    io.cprint(str(model))
    io.cprint(f"Using {torch.cuda.device_count()} GPUs")

    # Initialize optimizer
    if args.use_sgd:
        io.cprint("Using SGD optimizer")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        io.cprint("Using Adam optimizer")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Initialize learning rate scheduler
    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, 20, 0.5, args.epochs)

    # Loss function
    criterion = cal_loss

    # Training loop
    best_test_iou = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        
        for batch in train_loader:
            if args.dual_input:
                data1, data2, seg = batch
                data1, data2, seg = data1.to(device), data2.to(device), seg.to(device)
                data1 = data1[:,:,:args.num_features]
                data2 = data2[:,:,:args.num_features]
                data1 = data1.permute(0, 2, 1)
                data2 = data2.permute(0, 2, 1)
                batch_size = data1.size()[0]
                
                # Forward pass and optimization
                opt.zero_grad()
                seg_pred = model(data1, data2)
            else:
                data, seg = batch
                data, seg = data.to(device), seg.to(device)
                data = data[:,:,:args.num_features]
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                
                # Forward pass and optimization
                opt.zero_grad()
                seg_pred = model(data)
            
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, args.num_labels), seg.view(-1,1).squeeze())
            loss.backward()
            opt.step()
            
            # Get predictions
            pred = seg_pred.max(dim=2)[1]  # (batch_size, num_points)
            count += batch_size
            train_loss += loss.item() * batch_size
            
            # Prepare for metrics calculation
            seg_np = seg.cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            train_true_cls.append(seg_np.reshape(-1))
            train_pred_cls.append(pred_np.reshape(-1))
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
            
        # Update learning rate
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        
        # Calculate training metrics
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        train_ious = calculate_sem_IoU(train_pred_seg, train_true_seg)
        
        # Output training statistics
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, train iou: %.6f' % (
            epoch, 
            train_loss*1.0/count,
            train_acc,
            avg_per_class_acc,
            np.mean(train_ious)
        )
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_true_cls = []
        test_pred_cls = []
        test_true_seg = []
        test_pred_seg = []
        
        for batch in test_loader:
            if args.dual_input:
                data1, data2, seg = batch
                data1, data2, seg = data1.to(device), data2.to(device), seg.to(device)
                data1 = data1[:,:,:args.num_features]
                data2 = data2[:,:,:args.num_features]
                data1 = data1.permute(0, 2, 1)
                data2 = data2.permute(0, 2, 1)
                batch_size = data1.size()[0]
                
                # Forward pass
                seg_pred = model(data1, data2)
            else:
                data, seg = batch
                data, seg = data.to(device), seg.to(device)
                data = data[:,:,:args.num_features]
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                
                # Forward pass
                seg_pred = model(data)
            
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, args.num_labels), seg.view(-1,1).squeeze())
            pred = seg_pred.max(dim=2)[1]
            
            count += batch_size
            test_loss += loss.item() * batch_size
            
            # Prepare for metrics calculation
            seg_np = seg.cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            test_true_cls.append(seg_np.reshape(-1))
            test_pred_cls.append(pred_np.reshape(-1))
            test_true_seg.append(seg_np)
            test_pred_seg.append(pred_np)
            
        # Calculate test metrics
        test_true_cls = np.concatenate(test_true_cls)
        test_pred_cls = np.concatenate(test_pred_cls)
        test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        test_true_seg = np.concatenate(test_true_seg, axis=0)
        test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg)
        
        # Output test statistics
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (
            epoch,
            test_loss*1.0/count,
            test_acc,
            avg_per_class_acc,
            np.mean(test_ious)
        )
        io.cprint(outstr)
        
        # Save the best model
        if np.mean(test_ious) >= best_test_iou:
            best_test_iou = np.mean(test_ious)
            model_name = 'model_rdm' if args.dual_input else 'model'
            torch.save(model.state_dict(), 'outputs/%s/models/%s_%s.t7' % (args.exp_name, model_name, args.test_area))



if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Semantic Segmentation')
    parser.add_argument('--exp_name', type=str, default='exp', 
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', choices=['dgcnn'],
                        help='Model to use')
    parser.add_argument('--dual_input', action='store_true', default=False,
                        help='Use dual point cloud inputs with RDM Network')
    parser.add_argument('--secondary_data_type', type=str, default='ele_coherent_accumulated',
                        help='Type of secondary data to use (if different from primary)')
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='Size of training batch')
    parser.add_argument('--test_batch_size', type=int, default=1, 
                        help='Size of testing batch')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for data loading')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of epochs to train')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD optimizer instead of Adam')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, 
                        help='SGD momentum')
    parser.add_argument('--scheduler', type=str, default='cos', choices=['cos', 'step'],
                        help='LR scheduler to use')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='Disable CUDA training')
    parser.add_argument('--seed', type=int, default=1, 
                        help='Random seed')
    parser.add_argument('--num_points', type=int, default=4096,
                        help='Number of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, 
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, 
                        help='Number of nearest neighbors to use')
    parser.add_argument('--data_type', type=str, default='azi_pcd_normalthr_lq_accumulated',
                        help='Type of data to use')
    parser.add_argument('--num_features', type=int, default=6,
                        help='Number of input features per point')
    parser.add_argument('--num_labels', type=int, default=2,
                        help='Number of output classes')
    parser.add_argument('--scenes_list', type=str, 
                        default="'2023_05_18_17_22_18_cars_30s_1','2023_05_18_17_25_03_cars_30s_3'",
                        help='List of scenes to use for training and testing')
    parser.add_argument('--test_area', type=str, default='default',
                        help='Area for testing (used in model filename)')
    
    args = parser.parse_args()
    
    # Adjust parameters based on data type
    if "single" in args.data_type:
        args.num_points = 128
    
    # Initialize directories
    _init_()

    # Initialize logger
    io = IOStream('outputs/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    # Set up CUDA
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    # Start training
    train(args, io)


