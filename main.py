import argparse
import os
import shutil
import sys
import time
import warnings
import random
import itertools
import json
import datetime
from random import sample, seed
import csv
import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from pathlib import Path

from cgcnn.data import CIFData
from cgcnn.data import collate_pool, get_train_val_test_loader
from cgcnn.model import CrystalGraphConvNet

# Add logger utils import
from logger_utils import setup_logger, log_train_metrics, log_val_metrics

parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks for Multi-task Learning')
parser.add_argument('data_options', metavar='OPTIONS', nargs='+',
                    help='dataset options, started with the path to root dir, '
                         'then other options')
parser.add_argument('--normalize', action='store_true',
                    help='Enable normalization of target values (default: False)')
parser.add_argument('--task', choices=['regression', 'classification'],
                    default='regression', help='complete a regression or '
                                                   'classification task (default: regression)')
parser.add_argument('--task1-weight', default=0.5, type=float,
                   help='Weight for the first task (work function) loss (default: 0.5)')
parser.add_argument('--task2-weight', default=0.5, type=float,
                   help='Weight for the second task (band gap) loss (default: 0.5)')
parser.add_argument('--adaptive-weights', action='store_true',
                   help='Use adaptive task weights based on training losses (overrides task1-weight and task2-weight)')
parser.add_argument('--adaptive-weight-update', default=0.01, type=float,
                   help='Update rate for adaptive weights (default: 0.01)')
# Add conditional loss function related parameters
parser.add_argument('--conditional-loss', action='store_true',
                   help='Enable conditional band gap loss based on metal/semiconductor classification')
parser.add_argument('--classification-weight', default=0.6, type=float,
                   help='Weight for the metal/semiconductor classification loss (default: 0.3)')
parser.add_argument('--metal-weight', default=0.5, type=float,
                   help='Weight for metal band gap loss in conditional loss (default: 0.2)')
parser.add_argument('--semiconductor-weight', default=0.5, type=float,
                   help='Weight for semiconductor band gap loss in conditional loss (default: 0.8)')
# Add minimum band gap parameter
parser.add_argument('--min-bandgap', default=0.015, type=float,
                   help='Minimum band gap value for semiconductors (default: 0.015)')
# Add auxiliary task related parameters
parser.add_argument('--auxiliary-tasks', action='store_true',
                   help='Enable auxiliary learning tasks (electronegativity variance, formation energy, and first ionization energy)')
parser.add_argument('--aux-task1-weight', default=0.1, type=float,
                   help='Weight for auxiliary task 1 (electronegativity variance) loss (default: 0.1)')
parser.add_argument('--aux-task2-weight', default=0.1, type=float,
                   help='Weight for auxiliary task 2 (formation energy) loss (default: 0.1)')
parser.add_argument('--aux-task3-weight', default=0.1, type=float,
                   help='Weight for auxiliary task 3 (first ionization energy average) loss (default: 0.1)')
# Add material classification threshold parameter
parser.add_argument('--metal-threshold', default=0.5, type=float,
                   help='Threshold for classifying materials as metal based on metal probability (default: 0.5)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=80, type=int, metavar='N',
                    help='number of total epochs to run (default: 80)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default: '
                                       '0.01)')
parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int,
                    metavar='N', help='milestones for scheduler (default: '
                                      '[100])')
parser.add_argument('--scheduler', default='step', choices=['step', 'cosine'],
                    help='learning rate scheduler type: step (MultiStepLR) or cosine (CosineAnnealingLR) (default: step)')
parser.add_argument('--T-max', default=None, type=int,
                    help='maximum number of iterations for cosine annealing scheduler, defaults to epochs if not specified')
parser.add_argument('--eta-min', default=0, type=float,
                    help='minimum learning rate for cosine annealing scheduler (default: 0)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
train_group = parser.add_mutually_exclusive_group()
train_group.add_argument('--train-ratio', default=None, type=float, metavar='N',
                    help='number of training data to be loaded (default none)')
train_group.add_argument('--train-size', default=None, type=int, metavar='N',
                         help='number of training data to be loaded (default none)')
valid_group = parser.add_mutually_exclusive_group()
valid_group.add_argument('--val-ratio', default=0.1, type=float, metavar='N',
                    help='percentage of validation data to be loaded (default '
                         '0.1)')
valid_group.add_argument('--val-size', default=None, type=int, metavar='N',
                         help='number of validation data to be loaded (default '
                              '1000)')
test_group = parser.add_mutually_exclusive_group()
test_group.add_argument('--test-ratio', default=0.1, type=float, metavar='N',
                    help='percentage of test data to be loaded (default 0.1)')
test_group.add_argument('--test-size', default=None, type=int, metavar='N',
                        help='number of test data to be loaded (default 1000)')

parser.add_argument('--optim', default='SGD', type=str, metavar='SGD',
                    help='choose an optimizer, SGD or Adam, (default: SGD)')
parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')
parser.add_argument('--h-fea-len', default=128, type=int, metavar='N',
                    help='number of hidden features after pooling')
parser.add_argument('--n-conv', default=3, type=int, metavar='N',
                    help='number of conv layers')
parser.add_argument('--n-h', default=1, type=int, metavar='N',
                    help='number of hidden layers after pooling')
parser.add_argument('--pooling', default='mean', type=str, choices=['mean', 'attention'],
                    help='pooling method to aggregate atom features to crystal features (default: mean)')
parser.add_argument('--num-splits', default=1, type=int, metavar='N',
                    help='number of times to run with different seeds (default: 1)')

args = parser.parse_args(sys.argv[1:])

args.cuda = not args.disable_cuda and torch.cuda.is_available()

if args.task == 'regression':
    best_mae_error = 1e10
else:
    best_mae_error = 0.


def main():
    global args, best_mae_error
    
    # Setup result directories
    result_dir = Path("result")
    log_dir = result_dir / "log"
    predictions_dir = result_dir / "predictions"  # Add prediction results directory
    result_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    predictions_dir.mkdir(exist_ok=True)  # Create prediction results directory
    
    # Set base random seed
    base_seed = 123
    
    # Create run identifier (based on command line arguments)
    run_id = f"{args.scheduler}_{args.pooling}_{'adaptive' if args.adaptive_weights else 'fixed'}"
    
    # If conditional loss is enabled, add information to the identifier, including classification weight and semiconductor/metal weight ratio
    if args.conditional_loss:
        # Calculate the weight ratio of semiconductor to metal
        sc_metal_ratio = args.semiconductor_weight / args.metal_weight if args.metal_weight > 0 else 999.0
        run_id += f"_cond_cls{args.classification_weight:.2f}_ratio{sc_metal_ratio:.2f}_thresh{args.metal_threshold:.2f}"
    
    # Add second task weight
    run_id += f"_task2w{args.task2_weight:.2f}"
    
    # Explicitly specify auxiliary task status, whether enabled or disabled
    run_id += f"_aux{'+'if args.auxiliary_tasks else '-'}"
    
    # If auxiliary tasks are enabled, add weights for each auxiliary task
    if args.auxiliary_tasks:
        run_id += f"_aux1w{args.aux_task1_weight:.2f}_aux2w{args.aux_task2_weight:.2f}_aux3w{args.aux_task3_weight:.2f}"
    
    # Add identifiers for minimum band gap, learning rate, epochs, and weight decay
    run_id += f"_minbg{args.min_bandgap:.3f}_lr{args.lr}_e{args.epochs}_wd{float(args.weight_decay)}"
    
    print(f"\n{'='*80}\nRunning with configuration: {run_id}\n{'='*80}")
    
    # Number of runs
    num_splits = 1  # Default to run once
    if args.num_splits > 0:
        num_splits = args.num_splits
    print(f"Running {num_splits} {'time' if num_splits == 1 else 'times'} with different seeds")
    
    run_metrics = []
    split_timing_data = []  # Store timing data for each split
    
    # Load dataset and prepare collate function (only once outside the split loop)
    dataset = CIFData(*args.data_options)
    collate_fn = collate_pool
    
    # Load material types for conditional loss if needed
    material_types = {}
    if args.conditional_loss:
        print('Loading material types for metal/semiconductor classification...')
        material_types = load_material_types(args.data_options[0])
        print(f'Loaded {len(material_types)} material type entries')
    
    # Load auxiliary task data if needed
    auxiliary_data = {}
    if args.auxiliary_tasks:
        print('Loading auxiliary task data...')
        auxiliary_data = load_auxiliary_data(args.data_options[0])
        print(f'Loaded {len(auxiliary_data)} auxiliary data entries')
        
        # Print some sample auxiliary data for verification
        sample_keys = list(auxiliary_data.keys())[:5]
        print("Sample auxiliary data entries:")
        for key in sample_keys:
            print(f"  {key}: {auxiliary_data[key]}")
    
    # Run for each split
    for split in range(num_splits):
        # Set a different random seed for each run
        current_seed = base_seed + split
        torch.manual_seed(current_seed)
        torch.cuda.manual_seed(current_seed)
        np.random.seed(current_seed)
        seed(current_seed)
        
        print(f'--- Data Split {split + 1} (Seed: {current_seed}) ---')
        print(f'--- Using {args.pooling} pooling method ---')
        print(f'--- Using {args.scheduler} learning rate scheduler ---')
        if args.adaptive_weights:
            print(f'--- Using adaptive task weights (update rate: {args.adaptive_weight_update}) ---')
            print(f'--- Starting weights: task1={args.task1_weight:.4f}, task2={args.task2_weight:.4f} ---')
        else:
            print(f'--- Using fixed task weights: task1={args.task1_weight:.4f}, task2={args.task2_weight:.4f} ---')
        
        # Get data loaders for this split (with consistent random seed)
        train_loader, val_loader, test_loader = get_train_val_test_loader(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_size=args.batch_size,
            train_ratio=args.train_ratio,
            num_workers=args.workers,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            pin_memory=args.cuda,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            return_test=True)

        # Obtain target value normalizers for both tasks
        if args.task == 'classification':
            normalizer_task1 = Normalizer(torch.zeros(2))
            normalizer_task1.load_state_dict({'mean': 0., 'std': 1.})
            normalizer_task2 = Normalizer(torch.zeros(2))
            normalizer_task2.load_state_dict({'mean': 0., 'std': 1.})
        else:
            if len(dataset) < 500:
                warnings.warn('Dataset has less than 500 data points. '
                              'Lower accuracy is expected. ')
                sample_data_list = [dataset[i] for i in range(len(dataset))]
            else:
                sample_data_list = [dataset[i] for i in
                                    sample(range(len(dataset)), 500)]
            _, sample_targets, _ = collate_pool(sample_data_list)
            
            # Create separate normalizers for each task
            sample_target_task1 = sample_targets[:, 0].view(-1, 1)  # Work function
            sample_target_task2 = sample_targets[:, 1].view(-1, 1)  # Band gap
            
            normalizer_task1 = Normalizer(sample_target_task1, enable=args.normalize)
            normalizer_task2 = Normalizer(sample_target_task2, enable=args.normalize)
            ############################################################################2.28：17.23

        # Build model
        structures, _, _ = dataset[0]
        orig_atom_fea_len = structures[0].shape[-1]  #100
        nbr_fea_len = structures[1].shape[-1]  #Number of Gaussian basis
        model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                    atom_fea_len=args.atom_fea_len,
                                    n_conv=args.n_conv,
                                    h_fea_len=args.h_fea_len,
                                    n_h=args.n_h,
                                    classification=True if args.task == 'classification' else False,
                                    pooling_method=args.pooling)
        if args.cuda:
            model.cuda()

        # Define loss function and optimizer
        criterion = nn.MSELoss() if args.task == 'regression' else nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
        
        # Define learning rate scheduler
        if args.scheduler == 'step':
            scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)
            print(f"Using step learning rate scheduler with milestones at {args.lr_milestones}")
        elif args.scheduler == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            # If T_max is not specified, use the total number of epochs
            t_max = args.T_max if args.T_max is not None else args.epochs
            scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=args.eta_min)
            print(f"Using cosine annealing scheduler with T_max={t_max}, eta_min={args.eta_min}")
        else:
            raise ValueError(f"Unknown scheduler: {args.scheduler}")

        # Initialize task weights
        task1_weight = args.task1_weight
        task2_weight = args.task2_weight
        
        # Set up comprehensive logger
        logger = setup_logger(run_id, split, log_dir=str(log_dir))
        logger.info(f"Starting training run {run_id} split {split+1} (seed: {current_seed})")
        logger.info(f"Task weights - Work Function: {task1_weight:.6f}, Band Gap: {task2_weight:.6f}")
        
        # Log model configuration
        config_info = {
            "optimizer": args.optim,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "scheduler": args.scheduler,
            "conditional_loss": args.conditional_loss,
            "classification_weight": args.classification_weight if args.conditional_loss else 0,
            "metal_weight": args.metal_weight if args.conditional_loss else 0,
            "semiconductor_weight": args.semiconductor_weight if args.conditional_loss else 0,
            "auxiliary_tasks": args.auxiliary_tasks,
            "aux_task1_weight": args.aux_task1_weight if args.auxiliary_tasks else 0,
            "aux_task2_weight": args.aux_task2_weight if args.auxiliary_tasks else 0,
            "aux_task3_weight": args.aux_task3_weight if args.auxiliary_tasks else 0,
            "adaptive_weights": args.adaptive_weights,
            "adaptive_weight_update": args.adaptive_weight_update if args.adaptive_weights else 0
        }
        logger.info(f"Configuration: {json.dumps(config_info)}")
        
        # If using adaptive weights, initialize history
        if args.adaptive_weights:
            print("Using adaptive weights with update rate:", args.adaptive_weight_update)
            # Moving averages of the losses
            task1_loss_history = None
            task2_loss_history = None
            
            # Create legacy log file name with parameter combination info (for backward compatibility)
            legacy_log_filename = f"{run_id}_split{split+1}.log"
            legacy_log_path = log_dir / legacy_log_filename
            
            # Initialize weights log file for this split (legacy format)
            with open(legacy_log_path, 'w') as f:
                f.write(f"# Adaptive weights log for {run_id} split {split+1} (seed: {current_seed})\n")
                f.write("# Format: epoch, task1_weight, task2_weight, task1_loss, task2_loss\n")
                f.write(f"0, {task1_weight:.6f}, {task2_weight:.6f}, 0.0, 0.0\n")
        
        # Train and validate the model
        # Record training start time
        training_start_time = time.time()
        epoch_times = []
        
        for epoch in range(args.epochs):
            # Record start time for each epoch
            epoch_start_time = time.time()
            # Train with current weights
            train_metrics = train(
                train_loader, model, criterion, optimizer, epoch,
                normalizer_task1, normalizer_task2,
                task1_weight=task1_weight,
                task2_weight=task2_weight,
                material_types=material_types if args.conditional_loss else None,
                auxiliary_data=auxiliary_data if args.auxiliary_tasks else None,
                logger=logger
            )
            task1_loss, task2_loss = train_metrics['task1_loss'], train_metrics['task2_loss']


            # Update weights adaptively if option is enabled
            if args.adaptive_weights:
                # Initialize history on first epoch
                if epoch == 0:
                    task1_loss_history = task1_loss
                    task2_loss_history = task2_loss
                else:
                    # Update moving average
                    task1_loss_history = 0.9 * task1_loss_history + 0.1 * task1_loss
                    task2_loss_history = 0.9 * task2_loss_history + 0.1 * task2_loss
                
                # Calculate ratio between tasks
                if task1_loss_history > 0 and task2_loss_history > 0:
                    # Normalize losses relative to each other
                    total_loss = task1_loss_history + task2_loss_history
                    
                    # Update weights with smoothing factor to avoid large fluctuations
                    update_rate = args.adaptive_weight_update
                    task1_weight = (1 - update_rate) * task1_weight + update_rate * (task2_loss_history / total_loss)
                    task2_weight = (1 - update_rate) * task2_weight + update_rate * (task1_loss_history / total_loss)
                    
                    # Ensure weights sum to 1.0
                    weight_sum = task1_weight + task2_weight
                    task1_weight /= weight_sum
                    task2_weight /= weight_sum
                    
                    print(f'Adaptive weights updated: task1_weight={task1_weight:.4f}, task2_weight={task2_weight:.4f}')
                    
                    # Log weights for comprehensive logging
                    logger.info(f"Adaptive weights updated - Work Function: {task1_weight:.6f}, Band Gap: {task2_weight:.6f}")
                    
                    # Log weights for analysis - append to legacy log file in CSV format (for backward compatibility)
                    legacy_log_filename = f"{run_id}_split{split+1}.log"
                    legacy_log_path = log_dir / legacy_log_filename
                    with open(legacy_log_path, 'a') as f:
                        f.write(f"{epoch+1}, {task1_weight:.6f}, {task2_weight:.6f}, {task1_loss_history:.6f}, {task2_loss_history:.6f}\n")
            
            # Step the scheduler and print current learning rate
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Record end time for each epoch
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            epoch_times.append(epoch_duration)
            
            print(f'Epoch {epoch+1}/{args.epochs} completed. Time: {epoch_duration:.2f}s. Current learning rate: {current_lr:.6f}')
            
            val_metrics = validate(
                val_loader, model, criterion, normalizer_task1, normalizer_task2,
                material_types=material_types if args.conditional_loss else None,
                auxiliary_data=auxiliary_data if args.auxiliary_tasks else None,
                task1_weight=task1_weight, task2_weight=task2_weight,
                logger=logger, current_epoch=epoch
            )
            #val_mae1, val_r21 = val_metrics['wf_mae'], val_metrics['wf_r2']
            
        # Calculate training time statistics
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time
        avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0
        
        # Save timing statistics to file
        timing_filename = f"{run_id}_split{split+1}_timing.csv"
        timing_path = result_dir / timing_filename
        with open(timing_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value (seconds)', 'Value (minutes)', 'Value (hours)'])
            writer.writerow(['Total Training Time', f'{total_training_time:.2f}', f'{total_training_time/60:.2f}', f'{total_training_time/3600:.2f}'])
            writer.writerow(['Average Epoch Time', f'{avg_epoch_time:.2f}', f'{avg_epoch_time/60:.2f}', f'{avg_epoch_time/3600:.2f}'])
            writer.writerow(['Number of Epochs', args.epochs, '', ''])
            writer.writerow(['Fastest Epoch Time', f'{min(epoch_times):.2f}', f'{min(epoch_times)/60:.2f}', f'{min(epoch_times)/3600:.2f}'])
            writer.writerow(['Slowest Epoch Time', f'{max(epoch_times):.2f}', f'{max(epoch_times)/60:.2f}', f'{max(epoch_times)/3600:.2f}'])
            
            # Write detailed time for each epoch
            writer.writerow([])
            writer.writerow(['Epoch', 'Time (seconds)'])
            for i, epoch_time in enumerate(epoch_times, 1):
                writer.writerow([i, f'{epoch_time:.2f}'])
        
        print(f'Training completed! Total time: {total_training_time:.2f}s ({total_training_time/60:.2f}min)')
        print(f'Average epoch time: {avg_epoch_time:.2f}s')
        print(f'Timing statistics saved to: {timing_path}')
        
        # Test the model
        print('--- Evaluate Model on Test Set ---')
        logger.info('--- Evaluate Model on Test Set ---')
        
        # Call different validation logic based on whether conditional loss is enabled
        test_metrics = validate(
            test_loader, model, criterion, normalizer_task1, normalizer_task2,
            test=True,
            material_types=material_types if args.conditional_loss else None,
            auxiliary_data=auxiliary_data if args.auxiliary_tasks else None,
            task1_weight=task1_weight, task2_weight=task2_weight,
            logger=logger, current_epoch=epoch+1, phase="Test",
            run_id=run_id, predictions_dir=predictions_dir
        )
        if args.conditional_loss:
            test_mae1, test_r21 = test_metrics['wf_mae'], test_metrics['wf_r2']
            test_mae2, test_r22 = test_metrics['bg_mae'], test_metrics['bg_r2']
            test_cls_acc, test_cls_loss = test_metrics['cls_acc'], test_metrics['cls_loss']
            print(f'Task 1 (Work Function) - Test MAE: {test_mae1:.3f}, Test R^2: {test_r21:.3f}')
            print(f'Task 2 (Band Gap) - Test MAE: {test_mae2:.3f}, Test R^2: {test_r22:.3f}')
            print(f'Material Type Classification - Test Accuracy: {test_cls_acc:.3f}, Test Loss: {test_cls_loss:.3f}')
            logger.info(f"TEST SUMMARY - WF: MAE={test_mae1:.4f}, R^2={test_r21:.4f} | "
                        f"BG: MAE={test_mae2:.4f}, R^2={test_r22:.4f} | "
                        f"CLS: ACC={test_cls_acc:.4f}, Loss={test_cls_loss:.4f}")
            run_metrics.append((test_mae1, test_r21, test_mae2, test_r22, test_cls_acc, test_cls_loss))
        else:
            test_mae1, test_r21 = test_metrics['wf_mae'], test_metrics['wf_r2']
            test_mae2, test_r22 = test_metrics['bg_mae'], test_metrics['bg_r2']
            print(f'Task 1 (Work Function) - Test MAE: {test_mae1:.3f}, Test R^2: {test_r21:.3f}')
            print(f'Task 2 (Band Gap) - Test MAE: {test_mae2:.3f}, Test R^2: {test_r22:.3f}')
            logger.info(f"TEST SUMMARY - WF: MAE={test_mae1:.4f}, R^2={test_r21:.4f} | "
                        f"BG: MAE={test_mae2:.4f}, R^2={test_r22:.4f}")
            run_metrics.append((test_mae1, test_r21, test_mae2, test_r22))
            
        # Collect timing data for the current split
        split_timing_data.append({
            'split': split + 1,
            'total_time': total_training_time,
            'avg_epoch_time': avg_epoch_time,
            'num_epochs': args.epochs,
            'min_epoch_time': min(epoch_times),
            'max_epoch_time': max(epoch_times)
        })
    # Print final metrics
    print(f"Final Metrics for {run_id}:")
    for metric in run_metrics:
        print(f"  - {metric}")
        
    # Create summary timing statistics file
    if split_timing_data:
        overall_timing_filename = f"{run_id}_overall_timing_summary.csv"
        overall_timing_path = result_dir / overall_timing_filename
        
        # Calculate overall statistics
        total_experiment_time = sum(data['total_time'] for data in split_timing_data)
        avg_split_time = total_experiment_time / len(split_timing_data)
        overall_avg_epoch_time = sum(data['avg_epoch_time'] for data in split_timing_data) / len(split_timing_data)
        
        with open(overall_timing_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Experiment Timing Summary'])
            writer.writerow(['Run ID', run_id])
            writer.writerow(['Total Splits', len(split_timing_data)])
            writer.writerow(['Epochs per Split', args.epochs])
            writer.writerow([])
            
            writer.writerow(['Overall Statistics', 'Seconds', 'Minutes', 'Hours'])
            writer.writerow(['Total Experiment Time', f'{total_experiment_time:.2f}', f'{total_experiment_time/60:.2f}', f'{total_experiment_time/3600:.2f}'])
            writer.writerow(['Average Split Time', f'{avg_split_time:.2f}', f'{avg_split_time/60:.2f}', f'{avg_split_time/3600:.2f}'])
            writer.writerow(['Average Epoch Time (All Splits)', f'{overall_avg_epoch_time:.2f}', f'{overall_avg_epoch_time/60:.2f}', f'{overall_avg_epoch_time/3600:.2f}'])
            writer.writerow([])
            
            writer.writerow(['Split Details'])
            writer.writerow(['Split', 'Total Time (s)', 'Total Time (min)', 'Avg Epoch Time (s)', 'Min Epoch Time (s)', 'Max Epoch Time (s)'])
            for data in split_timing_data:
                writer.writerow([
                    data['split'],
                    f"{data['total_time']:.2f}",
                    f"{data['total_time']/60:.2f}",
                    f"{data['avg_epoch_time']:.2f}",
                    f"{data['min_epoch_time']:.2f}",
                    f"{data['max_epoch_time']:.2f}"
                ])
        
        print(f'Overall timing summary saved to: {overall_timing_path}')
        print(f'Total experiment time: {total_experiment_time:.2f}s ({total_experiment_time/60:.2f}min, {total_experiment_time/3600:.2f}h)')
        
    # Save the trained model with all necessary components
    model_filename = f"{run_id}_trained_model.pt"
    model_path = result_dir / model_filename
    
    # Create complete checkpoint with model, normalizers, and training args
    checkpoint = {
        'state_dict': model.state_dict(),
        'normalizer_task1': normalizer_task1.state_dict(),
        'normalizer_task2': normalizer_task2.state_dict(),
        'args': vars(args),  # Save training arguments
        'epoch': args.epochs,
        'best_mae_error': 0.0  # You might want to track this during training
    }
    
    torch.save(checkpoint, model_path)
    print(f"Saved trained model with normalizers to {model_path}")

    # Load the trained model
    model.load_state_dict(torch.load(model_path)['state_dict'])


    # Get a data loader for the entire dataset
    full_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.workers,
                                                collate_fn=collate_fn,
                                                pin_memory=args.cuda)

    # Write features to file
    feature_for_formula_path = f'feature_for_formula.csv'
    target_for_formula_path = f'target_for_formula.csv'

    # Write features to file (only for the last model in each combination)
    write_features_to_file(full_loader, model, feature_for_formula_path, target_for_formula_path)

    # Write metrics to a CSV file
    metrics_filename = f"{run_id}_metrics.csv"
    metrics_path = result_dir / metrics_filename
    print(f'Writing metrics to {metrics_path}...')
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Add different column headers based on whether material type classification is enabled
        if args.conditional_loss:
            writer.writerow(['WorkFunc_MAE', 'WorkFunc_R^2', 'BandGap_MAE', 'BandGap_R^2', 'MaterialType_ACC', 'MaterialType_Loss'])
        else:
            writer.writerow(['WorkFunc_MAE', 'WorkFunc_R^2', 'BandGap_MAE', 'BandGap_R^2'])
        writer.writerows(run_metrics)
    print('Metrics written to CSV file.')
    
    # Create a more comprehensive summary in mae.csv
    mae_filename = f"{run_id}_mae.csv"
    mae_path = result_dir / mae_filename
    with open(mae_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Create header with experiment configuration
        writer.writerow(['CGCNN Multi-Task Learning Results'])
        writer.writerow(['Configuration:', f'Pooling={args.pooling}', f'Scheduler={args.scheduler}',
                         f'Epochs={args.epochs}', f'Batch Size={args.batch_size}'])
        
        # Add task weight information
        if args.adaptive_weights:
            writer.writerow(['Task Weights:', 'Adaptive', f'Update Rate={args.adaptive_weight_update}'])
        else:
            writer.writerow(['Task Weights:', 'Fixed', f'WF={args.task1_weight:.3f}', f'BG={args.task2_weight:.3f}'])
        
        # Add conditional loss function information
        if args.conditional_loss:
            writer.writerow(['Conditional Loss:', f'Enabled', f'Classification Weight={args.classification_weight:.3f}',
                            f'Metal Weight={args.metal_weight:.3f}', f'Semiconductor Weight={args.semiconductor_weight:.3f}'])
        
        # Add empty row
        writer.writerow([])
        
        # Primary results table header
        if args.conditional_loss:
            writer.writerow(['Run #', 'WorkFunc_MAE', 'WorkFunc_R^2', 'BandGap_MAE', 'BandGap_R^2', 'MaterialType_ACC', 'MaterialType_Loss'])
        else:
            writer.writerow(['Run #', 'WorkFunc_MAE', 'WorkFunc_R^2', 'BandGap_MAE', 'BandGap_R^2'])
        
        # Write results for each run
        for i, metric in enumerate(run_metrics):
            if args.conditional_loss:
                writer.writerow([f'Run {i+1}', f'{metric[0]:.6f}', f'{metric[1]:.6f}', 
                               f'{metric[2]:.6f}', f'{metric[3]:.6f}', f'{metric[4]:.6f}', f'{metric[5]:.6f}'])
            else:
                writer.writerow([f'Run {i+1}', f'{metric[0]:.6f}', f'{metric[1]:.6f}', 
                               f'{metric[2]:.6f}', f'{metric[3]:.6f}'])
            
        # Calculate aggregate statistics
        avg_workfunc_mae = sum(metric[0] for metric in run_metrics) / len(run_metrics)
        avg_workfunc_r2 = sum(metric[1] for metric in run_metrics) / len(run_metrics)
        avg_bandgap_mae = sum(metric[2] for metric in run_metrics) / len(run_metrics) 
        avg_bandgap_r2 = sum(metric[3] for metric in run_metrics) / len(run_metrics)
        
        # Add empty row and averages
        writer.writerow([])
        if args.conditional_loss:
            avg_cls_acc = sum(metric[4] for metric in run_metrics) / len(run_metrics)
            avg_cls_loss = sum(metric[5] for metric in run_metrics) / len(run_metrics)
            writer.writerow(['Average', f'{avg_workfunc_mae:.6f}', f'{avg_workfunc_r2:.6f}', 
                           f'{avg_bandgap_mae:.6f}', f'{avg_bandgap_r2:.6f}',
                           f'{avg_cls_acc:.6f}', f'{avg_cls_loss:.6f}'])
        else:
            writer.writerow(['Average', f'{avg_workfunc_mae:.6f}', f'{avg_workfunc_r2:.6f}', 
                           f'{avg_bandgap_mae:.6f}', f'{avg_bandgap_r2:.6f}'])
        
        # Calculate and add standard deviations
        std_workfunc_mae = (sum((metric[0] - avg_workfunc_mae) ** 2 for metric in run_metrics) / len(run_metrics)) ** 0.5
        std_workfunc_r2 = (sum((metric[1] - avg_workfunc_r2) ** 2 for metric in run_metrics) / len(run_metrics)) ** 0.5
        std_bandgap_mae = (sum((metric[2] - avg_bandgap_mae) ** 2 for metric in run_metrics) / len(run_metrics)) ** 0.5
        std_bandgap_r2 = (sum((metric[3] - avg_bandgap_r2) ** 2 for metric in run_metrics) / len(run_metrics)) ** 0.5
        
        if args.conditional_loss:
            std_cls_acc = (sum((metric[4] - avg_cls_acc) ** 2 for metric in run_metrics) / len(run_metrics)) ** 0.5
            std_cls_loss = (sum((metric[5] - avg_cls_loss) ** 2 for metric in run_metrics) / len(run_metrics)) ** 0.5
            writer.writerow(['Std Dev', f'{std_workfunc_mae:.6f}', f'{std_workfunc_r2:.6f}', 
                           f'{std_bandgap_mae:.6f}', f'{std_bandgap_r2:.6f}',
                           f'{std_cls_acc:.6f}', f'{std_cls_loss:.6f}'])
        else:
            writer.writerow(['Std Dev', f'{std_workfunc_mae:.6f}', f'{std_workfunc_r2:.6f}', 
                           f'{std_bandgap_mae:.6f}', f'{std_bandgap_r2:.6f}'])
        
        # Add timestamp for reference
        import datetime
        writer.writerow([])
        writer.writerow(['Generated:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        
    print(f'MAE values including individual runs, averages, and standard deviations written to {mae_path}')
    
    # If adaptive weights were used, create a summary of weight evolution
    if args.adaptive_weights:
        print('Creating adaptive weights summary...')
        
        # Create summary file
        summary_filename = f"{run_id}_adaptive_weights_summary.csv"
        summary_path = result_dir / summary_filename
        with open(summary_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Split', 'Final Task1 Weight', 'Final Task2 Weight', 'Mean Task1 Weight', 'Mean Task2 Weight'])
            
            # Process each split log file
            for split in range(num_splits):
                try:
                    # Read the weights log file
                    log_filename = f"{run_id}_split{split+1}.log"
                    log_path = log_dir / log_filename
                    with open(log_path, 'r') as log_file:
                        lines = log_file.readlines()
                        
                        # Skip header lines
                        data_lines = [line for line in lines if not line.startswith('#')]
                        
                        # Extract weights
                        all_weights = []
                        for line in data_lines:
                            try:
                                parts = line.strip().split(',')
                                if len(parts) >= 3:  # Ensure we have enough columns
                                    task1_weight = float(parts[1])
                                    task2_weight = float(parts[2])
                                    all_weights.append((task1_weight, task2_weight))
                            except (ValueError, IndexError) as e:
                                print(f"Warning: Could not parse line: {line}")
                                continue
                        
                        if all_weights:
                            # Get final weights
                            final_task1_weight, final_task2_weight = all_weights[-1]
                            
                            # Calculate mean weights (excluding initial weights)
                            mean_task1_weight = sum(w[0] for w in all_weights[1:]) / len(all_weights[1:]) if len(all_weights) > 1 else all_weights[0][0]
                            mean_task2_weight = sum(w[1] for w in all_weights[1:]) / len(all_weights[1:]) if len(all_weights) > 1 else all_weights[0][1]
                            
                            writer.writerow([split+1, f'{final_task1_weight:.6f}', f'{final_task2_weight:.6f}', 
                                            f'{mean_task1_weight:.6f}', f'{mean_task2_weight:.6f}'])
                except Exception as e:
                    print(f"Error processing weights log for split {split+1}: {e}")
                    
        print(f'Adaptive weights summary written to {summary_path}')
    
    # End of this parameter combination run
    print(f"\nCompleted parameter combination: {run_id}\n{'='*80}")
    
    # If running multiple splits, calculate and save averages for this run
    if num_splits > 1:
        # Calculate averages across all splits
        avg_wf_mae = sum(metric[0] for metric in run_metrics) / len(run_metrics)
        avg_wf_r2 = sum(metric[1] for metric in run_metrics) / len(run_metrics)
        avg_bg_mae = sum(metric[2] for metric in run_metrics) / len(run_metrics)
        avg_bg_r2 = sum(metric[3] for metric in run_metrics) / len(run_metrics)
        
        print("\nAverage results across all splits:")
        print(f"Work Function - Avg MAE: {avg_wf_mae:.6f}, Avg R^2: {avg_wf_r2:.6f}")
        print(f"Band Gap - Avg MAE: {avg_bg_mae:.6f}, Avg R^2: {avg_bg_r2:.6f}")

    # Clean up feature files to avoid conflicts with future runs
    if os.path.exists('feature_for_formula.csv'):
        os.remove('feature_for_formula.csv')
    if os.path.exists('target_for_formula.csv'):
        os.remove('target_for_formula.csv')
    
    # Print completion message
    print("\nExperiment completed successfully.")
    print("Results are saved in the 'result' directory")
    print("Log files are saved in the 'result/log' directory")


def train(train_loader, model, criterion, optimizer, epoch, normalizer_task1, normalizer_task2, task1_weight=None, task2_weight=None, material_types=None, auxiliary_data=None, logger=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    task1_losses = AverageMeter()
    task2_losses = AverageMeter()
    
    # If conditional loss is enabled, add classification loss tracking
    if args.conditional_loss:
        classification_losses = AverageMeter()
        classification_accuracy = AverageMeter()
    
    # If auxiliary tasks are enabled, add auxiliary task loss tracking
    if args.auxiliary_tasks and auxiliary_data:
        aux_task1_losses = AverageMeter()  # Electronegativity variance
        aux_task2_losses = AverageMeter()  # Formation energy
        aux_task3_losses = AverageMeter()  # First ionization energy average
    
    # Use passed weights or default to args
    task1_weight = task1_weight if task1_weight is not None else args.task1_weight
    task2_weight = task2_weight if task2_weight is not None else args.task2_weight
    
    # Separate error tracking for each task
    if args.task == 'regression':
        mae_errors_task1 = AverageMeter()  # Work function
        mae_errors_task2 = AverageMeter()  # Band gap
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, targets, batch_cif_ids) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        if args.cuda:
            input_var = (Variable(input[0].cuda(non_blocking=True)),
                        Variable(input[1].cuda(non_blocking=True)),
                        input[2].cuda(non_blocking=True),
                        [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        else:
            input_var = (Variable(input[0]),
                        Variable(input[1]),
                        input[2],
                        input[3])
        
        # Separate targets for each task
        target_task1 = targets[:, 0].view(-1, 1)  # Work function
        target_task2 = targets[:, 1].view(-1, 1)  # Band gap
        
        # normalize targets for each task
        if args.task == 'regression':
            target_normed_task1 = normalizer_task1.norm(target_task1)
            target_normed_task2 = normalizer_task2.norm(target_task2)
        else:
            target_normed_task1 = target_task1.view(-1).long()
            target_normed_task2 = target_task2.view(-1).long()
            
        if args.cuda:
            target_var_task1 = Variable(target_normed_task1.cuda(non_blocking=True))
            target_var_task2 = Variable(target_normed_task2.cuda(non_blocking=True))
        else:
            target_var_task1 = Variable(target_normed_task1)
            target_var_task2 = Variable(target_normed_task2)

        # compute output for both tasks and material type classification
        # Add use_auxiliary_tasks parameter to ensure auxiliary tasks are only computed when enabled
        use_auxiliary_tasks = args.auxiliary_tasks and auxiliary_data is not None
        output_task1, output_task2, material_type_probs, atom_features_list, aux_outputs = model(
            *input_var, use_auxiliary_tasks=use_auxiliary_tasks)
        
        # Training phase: Band gap prediction proceeds normally, unaffected by classification
        # Post-processing mechanism is only applied during the testing phase
        
        # Calculate loss for both tasks
        loss_task1 = criterion(output_task1, target_var_task1)
        loss_task2 = criterion(output_task2, target_var_task2)
        
        # Initialize extra loss
        loss_classification = 0.0
        loss_aux_task1 = 0.0
        loss_aux_task2 = 0.0
        
        # Process material type classification (classification head needs to be trained regardless of whether conditional loss is enabled)
        if material_types:
            # Create metal/semiconductor batch masks and prepare classification targets
            material_type_targets = []
            
            for cif_id in batch_cif_ids:
                # Default to semiconductor (1) if not found in material_types
                material_type = material_types.get(cif_id, 1)
                material_type_targets.append(material_type)
            
            # Convert to tensor
            material_type_targets = torch.tensor(material_type_targets, dtype=torch.long)
            
            if args.cuda:
                material_type_targets = material_type_targets.cuda()
            
            # Calculate classification loss (cross-entropy)
            loss_classification = nn.CrossEntropyLoss()(material_type_probs, material_type_targets)
            
            # Calculate classification accuracy using an adjustable threshold
            # material_type_probs[:, 0] is the metal probability, material_type_probs[:, 1] is the semiconductor probability
            predicted_types = torch.argmax(material_type_probs, dim=1)
            correct = (predicted_types == material_type_targets).sum().item()
            classification_accuracy.update(correct / len(material_type_targets), len(material_type_targets))
            
            if args.conditional_loss:
                classification_losses.update(loss_classification.item(), len(material_type_targets))
        
        # Initialize auxiliary task loss variables
        loss_aux_task1 = torch.tensor(0.0)
        loss_aux_task2 = torch.tensor(0.0)
        loss_aux_task3 = torch.tensor(0.0)
        if args.cuda:
            loss_aux_task1 = loss_aux_task1.cuda()
            loss_aux_task2 = loss_aux_task2.cuda()
            loss_aux_task3 = loss_aux_task3.cuda()
            
        # If auxiliary tasks are enabled and the model returns auxiliary task outputs
        if args.auxiliary_tasks and auxiliary_data and aux_outputs is not None:
            # Extract auxiliary task targets
            aux_targets1 = []  # Electronegativity variance
            aux_targets2 = []  # Formation energy
            aux_targets3 = []  # First ionization energy average
            
            for cif_id in batch_cif_ids:
                # Remove possible file extension
                cif_id_clean = cif_id.split('.')[0]
                if cif_id_clean in auxiliary_data:
                    eneg_variance, formation_energy, ionization_energy = auxiliary_data[cif_id_clean]
                    aux_targets1.append(eneg_variance)
                    aux_targets2.append(formation_energy)
                    aux_targets3.append(ionization_energy)
                else:
                    # If no data, use default values
                    aux_targets1.append(0.0)
                    aux_targets2.append(0.0)
                    aux_targets3.append(0.0)
            
            # Convert to tensor
            aux_target1 = torch.tensor(aux_targets1, dtype=torch.float).view(-1, 1)
            aux_target2 = torch.tensor(aux_targets2, dtype=torch.float).view(-1, 1)
            aux_target3 = torch.tensor(aux_targets3, dtype=torch.float).view(-1, 1)
            
            if args.cuda:
                aux_target1 = aux_target1.cuda()
                aux_target2 = aux_target2.cuda()
                aux_target3 = aux_target3.cuda()
            
            # Use the auxiliary task outputs returned by the model
            batch_size = aux_target1.size(0)
            aux_output1, aux_output2, aux_output3 = aux_outputs
            
            # Calculate auxiliary task losses
            loss_aux_task1 = nn.MSELoss()(aux_output1, aux_target1)
            loss_aux_task2 = nn.MSELoss()(aux_output2, aux_target2)
            loss_aux_task3 = nn.MSELoss()(aux_output3, aux_target3)
            
            # Update statistics
            aux_task1_losses.update(loss_aux_task1.item(), batch_size)
            aux_task2_losses.update(loss_aux_task2.item(), batch_size)
            aux_task3_losses.update(loss_aux_task3.item(), batch_size)
        
        # Calculate total loss: main task loss + classification loss + auxiliary task loss
        loss = task1_weight * loss_task1
        
        # Calculate band gap loss based on conditional loss setting
        if args.conditional_loss and material_types:
            # Get predicted material types and probabilities
            material_type_probs_cpu = material_type_probs.cpu()
            metal_probs = material_type_probs_cpu[:, 0]
            sc_probs = material_type_probs_cpu[:, 1]
            
            # Calculate loss weights for metal and semiconductor samples separately
            metal_weight = args.metal_weight * metal_probs.mean()
            sc_weight = args.semiconductor_weight * sc_probs.mean()
            
            # Weighted calculation of band gap loss
            loss = loss + task2_weight * loss_task2 * (metal_weight + sc_weight)
        else:
            # Normal calculation when conditional loss is not enabled
            loss = loss + task2_weight * loss_task2
        
        # Add material classification loss (always included)
        if material_types:
            loss = loss + args.classification_weight * loss_classification
            
        # Add auxiliary task loss (if enabled)
        if args.auxiliary_tasks and auxiliary_data:
            aux_loss_contribution = args.aux_task1_weight * loss_aux_task1 + args.aux_task2_weight * loss_aux_task2 + args.aux_task3_weight * loss_aux_task3
            loss = loss + aux_loss_contribution

        # measure accuracy and record loss
        if args.task == 'regression':
            mae_error_task1 = mae(normalizer_task1.denorm(output_task1.data.cpu()), target_task1)
            
            # Get band gap MAE, apply different calculation methods based on conditional loss setting
            denorm_output_task2 = normalizer_task2.denorm(output_task2.data.cpu())
            
            if args.conditional_loss and material_types:
                # Get predicted material types and probabilities
                material_type_probs_cpu = material_type_probs.cpu()
                metal_probs = material_type_probs_cpu[:, 0].view(-1, 1)  # Metal probability (class 0)
                sc_probs = material_type_probs_cpu[:, 1].view(-1, 1)     # Semiconductor probability (class 1)
                
                # Calculate base MAE (without weights)
                base_mae = torch.abs(denorm_output_task2 - target_task2)
                
                # Weight MAE based on metal and semiconductor probabilities
                weighted_mae = base_mae * (metal_probs * args.metal_weight + sc_probs * args.semiconductor_weight)
                
                # Calculate average weighted MAE
                mae_error_task2 = weighted_mae.mean()
            else:
                # Normal calculation when conditional loss is not enabled
                mae_error_task2 = mae(denorm_output_task2, target_task2)
            
            # Update metrics
            losses.update(loss.data.cpu().item(), target_task1.size(0))
            task1_losses.update(loss_task1.data.cpu().item(), target_task1.size(0))
            task2_losses.update(loss_task2.data.cpu().item(), target_task2.size(0))
            mae_errors_task1.update(mae_error_task1.item(), target_task1.size(0))
            mae_errors_task2.update(mae_error_task2.item(), target_task2.size(0))
        else:
            # For classification tasks (not needed in this case)
            accuracy, precision, recall, fscore, auc_score = \
                class_eval(output_task1.data.cpu(), target_task1)
            losses.update(loss.data.cpu().item(), target_task1.size(0))
            accuracies.update(accuracy, target_task1.size(0))
            precisions.update(precision, target_task1.size(0))
            recalls.update(recall, target_task1.size(0))
            fscores.update(fscore, target_task1.size(0))
            auc_scores.update(auc_score, target_task1.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.task == 'regression':
                base_msg = 'Epoch: [{0}][{1}/{2}]\t' \
                           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                           'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                           'WF-MAE {mae1.val:.3f} ({mae1.avg:.3f})\t' \
                           'BG-MAE {mae2.val:.3f} ({mae2.avg:.3f})'.format(
                            epoch, i, len(train_loader), batch_time=batch_time,
                            data_time=data_time, loss=losses, 
                            mae1=mae_errors_task1, mae2=mae_errors_task2)
                
                # Add classification information (regardless of whether conditional loss is enabled)
                if material_types:
                    cls_msg = '\tClass-Acc: {acc:.3f}'.format(
                        acc=classification_accuracy.avg if classification_accuracy.count > 0 else 0
                    )
                    base_msg += cls_msg
                
                # Add conditional loss information
                if args.conditional_loss:
                    cond_msg = '\tClass-Loss: {cls:.3f}'.format(
                        cls=classification_losses.avg if classification_losses.count > 0 else 0
                    )
                    base_msg += cond_msg
                
                # Add auxiliary task information
                if args.auxiliary_tasks and auxiliary_data:
                    aux_msg = '\tAux: {aux1:.3f}/{aux2:.3f}/{aux3:.3f}'.format(
                        aux1=aux_task1_losses.avg if aux_task1_losses.count > 0 else 0,
                        aux2=aux_task2_losses.avg if aux_task2_losses.count > 0 else 0,
                        aux3=aux_task3_losses.avg if aux_task3_losses.count > 0 else 0
                    )
                    base_msg += aux_msg
                
                print(base_msg)
            else:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, accu=accuracies,
                    prec=precisions, recall=recalls, f1=fscores,
                    auc=auc_scores))
    
    # Collect all metrics in a dictionary
    metrics_dict = {
        'task1_loss': task1_losses.avg,
        'task2_loss': task2_losses.avg,
        'total_loss': losses.avg,
        'wf_mae': mae_errors_task1.avg,
        'bg_mae': mae_errors_task2.avg,
        'wf_weight': task1_weight,
        'bg_weight': task2_weight
    }
    
    # Add classification metrics if material_types is provided
    if material_types:
        metrics_dict.update({
            'cls_acc': classification_accuracy.avg
        })
        
        # Add conditional loss metrics if enabled
        if args.conditional_loss:
            metrics_dict.update({
                'cls_loss': classification_losses.avg if classification_losses.count > 0 else 0,
                'cls_weight': args.classification_weight
            })
    
    # Add auxiliary task metrics if enabled
    if args.auxiliary_tasks and auxiliary_data:
        metrics_dict.update({
            'aux1_mae': aux_task1_losses.avg if aux_task1_losses.count > 0 else 0,
            'aux2_mae': aux_task2_losses.avg if aux_task2_losses.count > 0 else 0,
            'aux3_mae': aux_task3_losses.avg if aux_task3_losses.count > 0 else 0,
            'aux1_weight': args.aux_task1_weight,
            'aux2_weight': args.aux_task2_weight,
            'aux3_weight': args.aux_task3_weight
        })
    
    # Log metrics to file if logger is provided
    if logger:
        log_train_metrics(logger, epoch + 1, metrics_dict)
    
    return metrics_dict

def validate(val_loader, model, criterion, normalizer_task1, normalizer_task2, test=False, material_types=None, auxiliary_data=None, task1_weight=None, task2_weight=None, logger=None, current_epoch=0, phase="Validation", run_id=None, predictions_dir=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    
    # Use passed weights or default to args
    task1_weight = task1_weight if task1_weight is not None else args.task1_weight
    task2_weight = task2_weight if task2_weight is not None else args.task2_weight
    
    # Separate tracking for each task
    if args.task == 'regression':
        mae_errors_task1 = AverageMeter()  # Work function
        mae_errors_task2 = AverageMeter()  # Band gap
        
    # Material classification tracker (regardless of whether conditional loss is enabled)
    classification_losses = AverageMeter()
    classification_accuracy = AverageMeter()
    
    # Separate trackers for auxiliary tasks
    if args.auxiliary_tasks and auxiliary_data:
        aux_task1_losses = AverageMeter()  # Electronegativity variance
        aux_task2_losses = AverageMeter()  # Formation energy
        aux_task3_losses = AverageMeter()  # First ionization energy average
        aux_task1_maes = AverageMeter()    # Electronegativity variance MAE
        aux_task2_maes = AverageMeter()    # Formation energy MAE
        aux_task3_maes = AverageMeter()    # First ionization energy average MAE
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()

    if test:
        test_targets_task1 = []
        test_preds_task1 = []
        test_targets_task2 = []
        test_preds_task2 = []
        test_cif_ids = []
        
        # Initialize lists for material type prediction results
        test_material_type_targets = []
        test_material_type_preds = []
        test_material_type_metal_probs = []
        test_material_type_sc_probs = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, targets, batch_cif_ids) in enumerate(val_loader):
        if args.cuda:
            with torch.no_grad():
                input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        else:
            with torch.no_grad():
                input_var = (Variable(input[0]),
                             Variable(input[1]),
                             input[2],
                             input[3])
                             
        # Separate targets for each task
        target_task1 = targets[:, 0].view(-1, 1)  # Work function
        target_task2 = targets[:, 1].view(-1, 1)  # Band gap
        
        # normalize targets
        if args.task == 'regression':
            target_normed_task1 = normalizer_task1.norm(target_task1)
            target_normed_task2 = normalizer_task2.norm(target_task2)
        else:
            target_normed_task1 = target_task1.view(-1).long()
            target_normed_task2 = target_task2.view(-1).long()
            
        if args.cuda:
            with torch.no_grad():
                target_var_task1 = Variable(target_normed_task1.cuda(non_blocking=True))
                target_var_task2 = Variable(target_normed_task2.cuda(non_blocking=True))
        else:
            with torch.no_grad():
                target_var_task1 = Variable(target_normed_task1)
                target_var_task2 = Variable(target_normed_task2)

        # compute output for both tasks and material type classification
        with torch.no_grad():
            use_auxiliary_tasks = args.auxiliary_tasks and auxiliary_data is not None
            output_task1, output_task2, material_type_probs, atom_features_list, aux_outputs = model(
                *input_var, use_auxiliary_tasks=use_auxiliary_tasks)
            
            # Apply post-processing mechanism only in the testing phase
            if test and args.task == 'regression':
                metal_probs = material_type_probs[:, 0]  # Metal probability
                metal_mask = (metal_probs > args.metal_threshold).float().view(-1, 1)  # Use adjustable threshold
                semiconductor_mask = (metal_probs <= args.metal_threshold).float().view(-1, 1)
                
                # Set band gap prediction to 0 for metal samples
                output_task2 = output_task2 * (1 - metal_mask)
                
                # For semiconductor samples, if the band gap prediction is less than or equal to min_bandgap, set it to min_bandgap
                low_bandgap_mask = (output_task2 <= args.min_bandgap).float()
                output_task2 = torch.where(
                    (semiconductor_mask * low_bandgap_mask).bool(),
                    torch.tensor(args.min_bandgap, device=output_task2.device, dtype=output_task2.dtype),
                    output_task2
                )
            
        # Calculate individual task losses
        loss_task1 = criterion(output_task1, target_var_task1)
        loss_task2 = criterion(output_task2, target_var_task2)
        
        # Initialize additional losses
        loss_classification = 0.0
        
        # Process material type classification (regardless of whether conditional loss is enabled)
        if material_types:
            # Get material type information for evaluation
            material_type_targets = []
            for cif_id in batch_cif_ids:
                material_type = material_types.get(cif_id, 1)  # Default to semiconductor (1)
                material_type_targets.append(material_type)
                
                # If in test mode, also collect prediction results
                if test and material_types:
                    test_material_type_targets.append(material_type)
            
            # Convert to tensor
            material_type_targets = torch.tensor(material_type_targets, dtype=torch.long)
            if args.cuda:
                material_type_targets = material_type_targets.cuda()
            
            # Calculate classification loss
            loss_classification = nn.CrossEntropyLoss()(material_type_probs, material_type_targets)
            classification_losses.update(loss_classification.item(), len(material_type_targets))
            
            # Calculate classification accuracy using an adjustable threshold
            # material_type_probs[:, 0] is the metal probability, material_type_probs[:, 1] is the semiconductor probability
            predicted_types = torch.argmax(material_type_probs, dim=1)
            correct = (predicted_types == material_type_targets).sum().item()
            classification_accuracy.update(correct / len(material_type_targets), len(material_type_targets))
            
            # If in test mode, collect material type prediction results
            if test and material_types:
                test_material_type_preds.extend(predicted_types.cpu().numpy().tolist())
                test_material_type_metal_probs.extend(material_type_probs[:, 0].cpu().numpy().tolist())
                test_material_type_sc_probs.extend(material_type_probs[:, 1].cpu().numpy().tolist())
        
        # Combined loss with primary tasks
        loss = task1_weight * loss_task1
        
   
        loss = loss + task2_weight * loss_task2 # Loss function should not enable conditional loss
        
        # Add classification loss if material_types is provided
        if material_types:
            loss = loss + args.classification_weight * loss_classification
            
        # Process auxiliary tasks if enabled
        if args.auxiliary_tasks and auxiliary_data:
            # Get material IDs for the current batch
            batch_cif_ids_list = list(batch_cif_ids)
            
            # Extract auxiliary task targets
            aux_targets1 = []  # Electronegativity variance
            aux_targets2 = []  # Formation energy
            aux_targets3 = []  # First ionization energy average
            
            for cif_id in batch_cif_ids_list:
                # Remove possible file extension
                cif_id_clean = cif_id.split('.')[0]
                if cif_id_clean in auxiliary_data:
                    eneg_variance, formation_energy, ionization_energy = auxiliary_data[cif_id_clean]
                    aux_targets1.append(eneg_variance)
                    aux_targets2.append(formation_energy)
                    aux_targets3.append(ionization_energy)
                else:
                    # If no data, use default values
                    aux_targets1.append(0.0)
                    aux_targets2.append(0.0)
                    aux_targets3.append(0.0)
            
            # Convert to tensor
            aux_target1 = torch.tensor(aux_targets1, dtype=torch.float).view(-1, 1)
            aux_target2 = torch.tensor(aux_targets2, dtype=torch.float).view(-1, 1)
            aux_target3 = torch.tensor(aux_targets3, dtype=torch.float).view(-1, 1)
            
            if args.cuda:
                aux_target1 = aux_target1.cuda()
                aux_target2 = aux_target2.cuda()
                aux_target3 = aux_target3.cuda()
            
            # Get auxiliary task outputs returned by the model
            batch_size = aux_target1.size(0)
            
            if aux_outputs is not None:
                aux_output1, aux_output2, aux_output3 = aux_outputs
                
                # Calculate auxiliary task loss and MAE
                loss_aux_task1 = nn.MSELoss()(aux_output1, aux_target1)
                loss_aux_task2 = nn.MSELoss()(aux_output2, aux_target2)
                loss_aux_task3 = nn.MSELoss()(aux_output3, aux_target3)
            
            # Calculate MAE for auxiliary tasks
            with torch.no_grad():
                if aux_outputs is not None:
                    aux_mae1 = mae(aux_output1.data.cpu(), aux_target1.cpu())
                    aux_mae2 = mae(aux_output2.data.cpu(), aux_target2.cpu())
                    aux_mae3 = mae(aux_output3.data.cpu(), aux_target3.cpu())
                else:
                    aux_mae1 = torch.tensor(0.0)
                    aux_mae2 = torch.tensor(0.0)
                    aux_mae3 = torch.tensor(0.0)
            
            # Update statistics (only when aux_outputs is available)
            if aux_outputs is not None:
                aux_task1_losses.update(loss_aux_task1.item(), batch_size)
                aux_task2_losses.update(loss_aux_task2.item(), batch_size)
                aux_task3_losses.update(loss_aux_task3.item(), batch_size)
                aux_task1_maes.update(aux_mae1, batch_size)
                aux_task2_maes.update(aux_mae2, batch_size)
                aux_task3_maes.update(aux_mae3, batch_size)
            
            # Add auxiliary tasks to total loss (only when aux_outputs is available)
            if aux_outputs is not None:
                loss = loss + args.aux_task1_weight * loss_aux_task1 + args.aux_task2_weight * loss_aux_task2 + args.aux_task3_weight * loss_aux_task3

        # measure accuracy and record loss
        if args.task == 'regression':
            # Ensure outputs are properly shaped
            output_task1 = output_task1.view(-1, 1)
            output_task2 = output_task2.view(-1, 1)
            
            # Calculate MAEs
            mae_error_task1 = mae(normalizer_task1.denorm(output_task1.data.cpu()), target_task1)
            
            # Get band gap MAE, apply different calculation methods based on conditional loss setting
            denorm_output_task2 = normalizer_task2.denorm(output_task2.data.cpu())
            

            mae_error_task2 = mae(denorm_output_task2, target_task2)
            
            # Update metrics
            losses.update(loss.data.cpu().item(), target_task1.size(0))
            mae_errors_task1.update(mae_error_task1.item(), target_task1.size(0))
            mae_errors_task2.update(mae_error_task2.item(), target_task2.size(0))

            if test:
                test_pred_task1 = normalizer_task1.denorm(output_task1.data.cpu())
                test_pred_task2 = normalizer_task2.denorm(output_task2.data.cpu())
                
                test_preds_task1 += test_pred_task1.view(-1).tolist()
                test_targets_task1 += target_task1.view(-1).tolist()
                
                test_preds_task2 += test_pred_task2.view(-1).tolist()
                test_targets_task2 += target_task2.view(-1).tolist()
                
                test_cif_ids += batch_cif_ids
        else:
            # For classification (not needed in this implementation)
            accuracy, precision, recall, fscore, auc_score = \
                class_eval(output_task1.data.cpu(), target_task1)
            losses.update(loss.data.cpu().item(), target_task1.size(0))
            accuracies.update(accuracy, target_task1.size(0))
            precisions.update(precision, target_task1.size(0))
            recalls.update(recall, target_task1.size(0))
            fscores.update(fscore, target_task1.size(0))
            auc_scores.update(auc_score, target_task1.size(0))
            if test:
                test_pred = torch.exp(output_task1.data.cpu())
                test_target = target_task1
                assert test_pred.shape[1] == 2
                test_preds_task1 += test_pred[:, 1].tolist()
                test_targets_task1 += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.task == 'regression':
                if material_types:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'WF-MAE {mae1.val:.3f} ({mae1.avg:.3f})\t'
                          'BG-MAE {mae2.val:.3f} ({mae2.avg:.3f})\t'
                          'MT-ACC {cls_acc.val:.3f} ({cls_acc.avg:.3f})\t'
                          'MT-Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        mae1=mae_errors_task1, mae2=mae_errors_task2,
                        cls_acc=classification_accuracy, cls_loss=classification_losses))
                else:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'WF-MAE {mae1.val:.3f} ({mae1.avg:.3f})\t'
                          'BG-MAE {mae2.val:.3f} ({mae2.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        mae1=mae_errors_task1, mae2=mae_errors_task2))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    accu=accuracies, prec=precisions, recall=recalls,
                    f1=fscores, auc=auc_scores))

    if test:
        star_label = '**'
        import csv
        
        # Write detailed prediction results to the new predictions directory
        if predictions_dir is not None and run_id is not None:
            # Create a filename that includes configuration information
            prediction_filename = f"{run_id}_predictions.csv"
            prediction_path = predictions_dir / prediction_filename
            
            # Write test results, including all relevant information
            with open(prediction_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['cif_id', 'work_function_target', 'work_function_pred', 
                                 'band_gap_target', 'band_gap_pred', 
                                 'material_type_target', 'material_type_pred', 
                                 'metal_prob', 'semiconductor_prob'])
                
                for idx, (cif_id, target1, pred1, target2, pred2) in enumerate(zip(
                    test_cif_ids, test_targets_task1, test_preds_task1, 
                    test_targets_task2, test_preds_task2)):
                    
                    # Get material type information
                    if material_types:
                        # Get material type target value
                        material_type_target = material_types.get(cif_id, 1)  # Default to semiconductor (1)
                        
                        # Get material type prediction value and probability
                        if idx < len(test_material_type_preds):
                            material_type_pred = test_material_type_preds[idx]
                            metal_prob = test_material_type_metal_probs[idx]
                            semiconductor_prob = test_material_type_sc_probs[idx]
                        else:
                            material_type_pred = ""
                            metal_prob = ""
                            semiconductor_prob = ""
                        
                        writer.writerow([cif_id, target1, pred1, target2, pred2, 
                                        material_type_target, material_type_pred,
                                        metal_prob, semiconductor_prob])
                    else:
                        # If there is no material type data, still keep the format consistent
                        writer.writerow([cif_id, target1, pred1, target2, pred2, 
                                        "", "", "", ""])
            
            print(f"Detailed predictions saved to: {prediction_path}")
        
        # Write test results
        if material_types:
            # Verify if material type data exists
            has_mt_data = (len(test_material_type_targets) > 0 and
                         len(test_material_type_preds) > 0)
            
            # Write test results, including material type information
            with open('test_results.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['cif_id', 'work_function_target', 'work_function_pred', 
                                 'band_gap_target', 'band_gap_pred', 
                                 'material_type_target', 'material_type_pred', 
                                 'metal_prob', 'semiconductor_prob'])
                
                for idx, (cif_id, target1, pred1, target2, pred2) in enumerate(zip(
                    test_cif_ids, test_targets_task1, test_preds_task1, 
                    test_targets_task2, test_preds_task2)):
                    
                    # Get material type information
                    if has_mt_data and idx < len(test_material_type_targets):
                        material_type_target = test_material_type_targets[idx]
                        material_type_pred = test_material_type_preds[idx]
                        metal_prob = test_material_type_metal_probs[idx]
                        semiconductor_prob = test_material_type_sc_probs[idx]
                        
                        writer.writerow((cif_id, target1, pred1, target2, pred2, 
                                        material_type_target, material_type_pred,
                                        metal_prob, semiconductor_prob))
                    else:
                        material_type = material_types.get(cif_id, 1)  # Default to semiconductor (1)
                        writer.writerow((cif_id, target1, pred1, target2, pred2, 
                                        material_type, "", "", ""))
        else:
            with open('test_results.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['cif_id', 'work_function_target', 'work_function_pred', 'band_gap_target', 'band_gap_pred'])
                for cif_id, target1, pred1, target2, pred2 in zip(
                    test_cif_ids, test_targets_task1, test_preds_task1, test_targets_task2, test_preds_task2):
                    writer.writerow((cif_id, target1, pred1, target2, pred2))
        
        # Calculate R^2 scores for both tasks
        test_r2_task1 = metrics.r2_score(test_targets_task1, test_preds_task1)
        test_r2_task2 = metrics.r2_score(test_targets_task2, test_preds_task2)

        print_msg = ' {star} WF-MAE {mae1.avg:.3f}, WF-R^2 {r2_1:.3f}, BG-MAE {mae2.avg:.3f}, BG-R^2 {r2_2:.3f}'.format(
            star=star_label, mae1=mae_errors_task1, r2_1=test_r2_task1, 
            mae2=mae_errors_task2, r2_2=test_r2_task2)
            
        if material_types:
            print_msg += ', MT-ACC {acc.avg:.3f}, MT-Loss {loss.avg:.3f}'.format(
                acc=classification_accuracy, loss=classification_losses)
        
        print(print_msg)
        
        # Collect metrics in dictionary
        metrics_dict = {
            'wf_mae': mae_errors_task1.avg,
            'bg_mae': mae_errors_task2.avg,
            'wf_r2': test_r2_task1,
            'bg_r2': test_r2_task2,
            'total_loss': losses.avg,
            'wf_weight': task1_weight,
            'bg_weight': task2_weight
        }
        
        # Add classification metrics if material_types is provided
        if material_types:
            metrics_dict.update({
                'cls_loss': classification_losses.avg,
                'cls_acc': classification_accuracy.avg,
                'cls_weight': args.classification_weight
            })
            
        # Add auxiliary task metrics if enabled
        if args.auxiliary_tasks and auxiliary_data and 'aux_task1_maes' in locals():
            metrics_dict.update({
                'aux1_loss': aux_task1_losses.avg if aux_task1_losses.count > 0 else 0,
                'aux2_loss': aux_task2_losses.avg if aux_task2_losses.count > 0 else 0,
                'aux3_loss': aux_task3_losses.avg if aux_task3_losses.count > 0 else 0,
                'aux1_mae': aux_task1_maes.avg if aux_task1_maes.count > 0 else 0,
                'aux2_mae': aux_task2_maes.avg if aux_task2_maes.count > 0 else 0,
                'aux3_mae': aux_task3_maes.avg if aux_task3_maes.count > 0 else 0,
                'aux1_weight': args.aux_task1_weight,
                'aux2_weight': args.aux_task2_weight,
                'aux3_weight': args.aux_task3_weight
            })
        
        # Log metrics to file if logger is provided
        if logger:
            log_val_metrics(logger, current_epoch, metrics_dict, phase="Test")
            
        return metrics_dict
    
    else:
        star_label = '*'
        
        # Collect metrics in dictionary
        metrics_dict = {
            'wf_mae': mae_errors_task1.avg,
            'bg_mae': mae_errors_task2.avg,
            'total_loss': losses.avg,
            'wf_weight': task1_weight,
            'bg_weight': task2_weight
        }
        
        if args.task == 'regression':
            if material_types:
                print(' {star} WF-MAE {mae1.avg:.3f}, BG-MAE {mae2.avg:.3f}, MT-ACC {cls_acc.avg:.3f}, MT-Loss {cls_loss.avg:.4f}'.format(
                    star=star_label, mae1=mae_errors_task1, mae2=mae_errors_task2,
                    cls_acc=classification_accuracy, cls_loss=classification_losses))
                
                # Add classification metrics
                metrics_dict.update({
                    'cls_loss': classification_losses.avg,
                    'cls_acc': classification_accuracy.avg,
                    'cls_weight': args.classification_weight
                })
            else:
                print(' {star} WF-MAE {mae1.avg:.3f}, BG-MAE {mae2.avg:.3f}'.format(
                    star=star_label, mae1=mae_errors_task1, mae2=mae_errors_task2))
                
            # Add auxiliary task metrics if enabled
            if args.auxiliary_tasks and auxiliary_data and 'aux_task1_maes' in locals():
                metrics_dict.update({
                    'aux1_loss': aux_task1_losses.avg if aux_task1_losses.count > 0 else 0,
                    'aux2_loss': aux_task2_losses.avg if aux_task2_losses.count > 0 else 0,
                    'aux3_loss': aux_task3_losses.avg if aux_task3_losses.count > 0 else 0,
                    'aux1_mae': aux_task1_maes.avg if aux_task1_maes.count > 0 else 0,
                    'aux2_mae': aux_task2_maes.avg if aux_task2_maes.count > 0 else 0,
                    'aux3_mae': aux_task3_maes.avg if aux_task3_maes.count > 0 else 0,
                    'aux1_weight': args.aux_task1_weight,
                    'aux2_weight': args.aux_task2_weight,
                    'aux3_weight': args.aux_task3_weight
                })
        else:
            print(' {star} AUC {auc.avg:.3f}'.format(star=star_label, auc=auc_scores))
            metrics_dict['auc'] = auc_scores.avg
        
        # Log metrics to file if logger is provided
        if logger:
            log_val_metrics(logger, current_epoch, metrics_dict)
            
        return metrics_dict


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor, enable=True):
        """tensor is taken as a sample to calculate the mean and std"""
        self.enable = enable
        if enable:
            self.mean = torch.mean(tensor)
            self.std = torch.std(tensor)
        else:
            self.mean = 0
            self.std = 1

    def norm(self, tensor):
        if self.enable:
            return (tensor - self.mean) / self.std
        else:
            return tensor

    def denorm(self, normed_tensor):
        if self.enable:
            return normed_tensor * self.std + self.mean
        else:
            return normed_tensor

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']



def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = args.lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def load_material_types(root_dir):
    """
    Load material types (metal: 0, semiconductor: 1) from all.csv file
    
    Parameters:
    root_dir (str): Path to the root directory containing all.csv
    
    Returns:
    dict: Dictionary mapping material IDs to their types (0: metal, 1: semiconductor)
    """
    material_types = {}
    all_csv_path = os.path.join(root_dir, "all.csv")
    
    if os.path.exists(all_csv_path):
        with open(all_csv_path, 'r') as f:
            reader = csv.reader(f)
            # Skip header row (if any)
            try:
                header = next(reader, None)
                for row in reader:
                    if len(row) >= 2:
                        material_id = row[0]
                        # Assume the second column is material type: 0 for metal, 1 for semiconductor
                        try:
                            material_type = int(row[1])
                            material_types[material_id] = material_type
                        except (ValueError, IndexError):
                            # Default to semiconductor (1) if conversion fails
                            material_types[material_id] = 1
            except Exception as e:
                print(f"Error loading material types from {all_csv_path}: {e}")
                return {}
    else:
        print(f"Warning: Material types file {all_csv_path} not found")
        
    return material_types


def load_auxiliary_data(root_dir):
    """
    Load auxiliary task data (electronegativity variance, formation energy, first ionization energy average)
    
    Parameters:
    root_dir (str): Path to the root directory containing ax.csv
    
    Returns:
    dict: Dictionary mapping material IDs to auxiliary data tuples (eneg_variance, formation_energy, ionization_energy)
    """
    auxiliary_data = {}
    aux_data_path = os.path.join(root_dir, "ax.csv")
    
    if os.path.exists(aux_data_path):
        with open(aux_data_path, 'r') as f:
            reader = csv.reader(f)
            # Skip header
            try:
                header = next(reader, None)
                
                # Read auxiliary data
                for row in reader:
                    if len(row) >= 6:  # Make sure the row has all necessary columns
                        material_id = row[0]  # First column is id
                        try:
                            eneg_variance = float(row[3])  # Fourth column is electronegativity_variance
                            formation_energy = float(row[4])  # Fifth column is formation_energy
                            ionization_energy = float(row[5])  # Sixth column is first_ionization_energy_avg
                            auxiliary_data[material_id] = (eneg_variance, formation_energy, ionization_energy)
                        except (ValueError, IndexError):
                            # Skip if conversion fails
                            continue
            except Exception as e:
                print(f"Error loading auxiliary data from {aux_data_path}: {e}")
                return {}
        
        print(f"Loaded {len(auxiliary_data)} auxiliary task data entries from {aux_data_path}")
    else:
        print(f"Warning: Auxiliary data file {aux_data_path} not found")
        
    return auxiliary_data
def write_features_to_file(data_loader, model, feature_file_path, result_file_path):
    # switch to evaluate mode
    model.eval()

    with open(feature_file_path, 'w') as feature_file, open(result_file_path, 'w') as result_file:
        feature_writer = csv.writer(feature_file)
        result_writer = csv.writer(result_file)
        result_writer.writerow(['CIF ID', 'WorkFunc_Target', 'WorkFunc_Prediction', 'BandGap_Target', 'BandGap_Prediction'])

        for i, (input, targets, cif_ids) in enumerate(data_loader):
            if args.cuda:
                with torch.no_grad():
                    input_var = (Variable(input[0].cuda(non_blocking=True)),
                                 Variable(input[1].cuda(non_blocking=True)),
                                 input[2].cuda(non_blocking=True),
                                 [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
            else:
                with torch.no_grad():
                    input_var = (Variable(input[0]),
                                 Variable(input[1]),
                                 input[2],
                                 input[3])

            # Separate targets for each task
            target_task1 = targets[:, 0].view(-1, 1)  # Work function
            target_task2 = targets[:, 1].view(-1, 1)  # Band gap

            # compute output for both tasks
            with torch.no_grad():
                # Pass use_auxiliary_tasks=False as this is prediction only
                output_task1, output_task2, material_type_probs, atom_features_list, _ = model(*input_var, use_auxiliary_tasks=False)

                # For materials predicted as metal, set the band gap prediction to 0 (using an adjustable threshold)
                # For materials predicted as semiconductor, if the band gap prediction is less than or equal to min_bandgap, set it to min_bandgap
                if args.task == 'regression':
                    metal_probs = material_type_probs[:, 0]  # Metal probability
                    metal_mask = (metal_probs > args.metal_threshold).float().view(-1, 1)  # Use adjustable threshold
                    semiconductor_mask = (metal_probs <= args.metal_threshold).float().view(-1, 1)
                    
                    # Set band gap prediction for metal samples to 0
                    output_task2 = output_task2 * (1 - metal_mask)
                    
                    # For semiconductor samples, if the band gap prediction is less than or equal to min_bandgap, set it to min_bandgap
                    low_bandgap_mask = (output_task2 <= args.min_bandgap).float()
                    output_task2 = torch.where(
                        (semiconductor_mask * low_bandgap_mask).bool(),
                        torch.tensor(args.min_bandgap, device=output_task2.device, dtype=output_task2.dtype),
                        output_task2
                    )

            # write features (atom contributions)
            for tensor in atom_features_list:
                feature_vector = tensor.detach().cpu().numpy().tolist()
                line = ','.join(map(str, feature_vector))
                feature_file.write(line + '\n')

            # write predictions for both tasks
            pred_task1 = output_task1.detach().cpu().numpy()
            pred_task2 = output_task2.detach().cpu().numpy()
            
            for cif_id, t1, p1, t2, p2 in zip(
                cif_ids, 
                target_task1.cpu().numpy(), 
                pred_task1,
                target_task2.cpu().numpy(), 
                pred_task2
            ):
                result_writer.writerow([cif_id, t1.item(), p1.item(), t2.item(), p2.item()])

if __name__ == '__main__':
    main()
