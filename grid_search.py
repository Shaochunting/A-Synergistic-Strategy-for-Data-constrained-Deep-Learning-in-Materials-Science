import subprocess
import itertools
import os
import argparse
import csv
import glob
from pathlib import Path
import time
from datetime import datetime

def run_grid_search(data_dir, num_splits=1, epochs=80):
    """
                     print(f"Results for {run_id}:")
                    print(f"  Learning Rate: {lr_value}, Epochs: {epoch_value}, Weight Decay: {wd_value}")
                    print(f"  Classification Weight: {cls_weight}, SC-Metal Ratio: {sc_metal_ratio}")
                    print(f"  Metal Weight: {metal_weight:.4f}, SC Weight: {semiconductor_weight:.4f}")
                    print(f"  Work Function: MAE = {avg_wf_mae:.6f}, R^2 = {avg_wf_r2:.6f}")
                    print(f"  Band Gap: MAE = {avg_bg_mae:.6f}, R^2 = {avg_bg_r2:.6f}")
                    if avg_mt_acc is not None:
                        print(f"  Material Type: ACC = {avg_mt_acc:.6f}, Loss = {avg_mt_loss:.6f}") a grid search by calling main.py with different parameter combinations
    
    Parameters:
    -----------
    data_dir : str
        Path to the data directory
    num_splits : int
        Number of runs per parameter combination
    epochs : int  
        Number of epochs for training
    """
    # Setup result directories
    result_dir = Path("result")
    log_dir = result_dir / "log"
    result_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
            # ============================================================================
        # Parameter configuration area - modify the parameters you want to explore here
        # ============================================================================
    
        # Note: if a parameter has only one value it will remain fixed
        #       If a parameter has multiple values, all combinations will be explored

    scheduler_options = ['step']     # learning rate scheduler
    pooling_options = ['attention']            # pooling method (fixed to 'attention')
    adaptive_weight_options = [True]   # adaptive weights
    conditional_loss_options = [True]  # conditional loss function
    classification_weight_options = [0.2]  # classification weight
    sc_metal_ratio_options = [1.1]  # ratio of semiconductor to metal weight
    metal_threshold_options = [0.7]  # metal classification threshold
    task2_weight_options = [0.2]  # second task (band gap) weight
    auxiliary_tasks_options = [True]   # auxiliary tasks
    aux_task1_weight_options = [0.11]  # aux task1 weight (electronegativity variance)
    aux_task2_weight_options = [0.11]  # aux task2 weight (formation energy)
    aux_task3_weight_options = [0.05]  # aux task3 weight (first ionization energy mean)
    min_bandgap_options = [0.015]  # minimum semiconductor bandgap value
    learning_rate_options = [0.011]     # learning rate
    epochs_options = [12]                     # training epochs (fixed to 80 in help by default)
    weight_decay_options = [0.0] # weight decay

    # ============================================================================
    # End of parameter configuration
    # ============================================================================
    # Get all combinations
    param_combinations = list(itertools.product(
        scheduler_options, pooling_options, adaptive_weight_options,
        conditional_loss_options, classification_weight_options, sc_metal_ratio_options, metal_threshold_options,
        task2_weight_options, auxiliary_tasks_options, aux_task1_weight_options, aux_task2_weight_options, aux_task3_weight_options,
        min_bandgap_options, learning_rate_options, epochs_options, weight_decay_options
    ))
    
    # Display parameter configuration information
    print("=" * 80)
    print("Current parameter configuration:")
    print(f"  Scheduler: {scheduler_options} ({len(scheduler_options)} options)")
    print(f"  Pooling: {pooling_options} ({len(pooling_options)} options)")
    print(f"  Adaptive Weights: {adaptive_weight_options} ({len(adaptive_weight_options)} options)")
    print(f"  Conditional Loss: {conditional_loss_options} ({len(conditional_loss_options)} options)")
    print(f"  Classification Weight: {classification_weight_options} ({len(classification_weight_options)} options)")
    print(f"  SC-Metal Ratio: {sc_metal_ratio_options} ({len(sc_metal_ratio_options)} options)")
    print(f"  Metal Threshold: {metal_threshold_options} ({len(metal_threshold_options)} options)")
    print(f"  Task2 Weight: {task2_weight_options} ({len(task2_weight_options)} options)")
    print(f"  Auxiliary Tasks: {auxiliary_tasks_options} ({len(auxiliary_tasks_options)} options)")
    print(f"  Aux Task1 Weight: {aux_task1_weight_options} ({len(aux_task1_weight_options)} options)")
    print(f"  Aux Task2 Weight: {aux_task2_weight_options} ({len(aux_task2_weight_options)} options)")
    print(f"  Aux Task3 Weight: {aux_task3_weight_options} ({len(aux_task3_weight_options)} options)")
    print(f"  Min Bandgap: {min_bandgap_options} ({len(min_bandgap_options)} options)")
    print(f"  Learning Rate: {learning_rate_options} ({len(learning_rate_options)} options)")
    print(f"  Epochs: {epochs_options} ({len(epochs_options)} options)")
    print(f"  Weight Decay: {weight_decay_options} ({len(weight_decay_options)} options)")
    print("-" * 80)
    print(f"Total combinations: {len(param_combinations)}")
    print(f"Each combination will run {num_splits} times")
    print(f"Estimated total experiment runs: {len(param_combinations) * num_splits}")
    print("=" * 80)
    
    print(f"Exploring {len(param_combinations)} parameter combinations:")
    for i, combo in enumerate(param_combinations):
        sched, pool, adaptive, cond, cls_weight, sc_metal_ratio, metal_thresh, task2_weight, aux, aux1_weight, aux2_weight, aux3_weight, min_bandgap, lr, epoch_val, wd = combo
        print(f"Combination {i+1}: Scheduler={sched}, Pooling={pool}, "
              f"Adaptive={adaptive}, Conditional={cond}, Classification Weight={cls_weight}, "
              f"SC-Metal Ratio={sc_metal_ratio}, Metal Threshold={metal_thresh}, Task2 Weight={task2_weight}, "
              f"Auxiliary={aux}, Aux Weights={aux1_weight}-{aux2_weight}-{aux3_weight}, "
              f"Min Bandgap={min_bandgap}, LR={lr}, Epochs={epoch_val}, WD={wd}")
    
    # Display total number of combinations
    print(f"\nAll combinations will use specified learning rates, epochs, and weight decay values")
    
    # Generate a timestamp for this grid search run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Summary file for all combinations
    summary_all_path = result_dir / f"all_combinations_summary_{timestamp}.csv"
    with open(summary_all_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Scheduler', 'Pooling', 'Adaptive Weights', 'Conditional Loss', 
                        'Classification Weight', 'SC-Metal Ratio', 'Metal Threshold', 'Metal Weight', 'SC Weight', 
                        'Task2 Weight', 'Aux Tasks', 'Aux Task1 Weight', 'Aux Task2 Weight', 'Aux Task3 Weight',
                        'Min Bandgap', 'Learning Rate', 'Epochs', 'Weight Decay',
                        'Avg WF-MAE', 'Avg WF-R^2', 'Avg BG-MAE', 'Avg BG-R^2', 
                        'Material Type ACC', 'Material Type Loss'])  # Added columns for new parameters    # Run each combination
    for combo in param_combinations:
            # Parse parameter combination
        sched, pool, adaptive, cond, cls_weight, sc_metal_ratio, metal_thresh, task2_weight, aux, aux1_weight, aux2_weight, aux3_weight, min_bandgap, lr_value, epoch_value, wd_value = combo
            
    # Compute metal and semiconductor weights so they sum to 1
    # Formula: sc_weight = sc_metal_ratio / (1 + sc_metal_ratio), metal_weight = 1 - sc_weight
        semiconductor_weight =2* sc_metal_ratio / (1 + sc_metal_ratio)
        metal_weight = 2 - semiconductor_weight
            
        # Create run identifier
        run_id = f"{sched}_{pool}_{'adaptive' if adaptive else 'fixed'}"
        
    # Add identifiers for conditional bandgap, classification weight, and semiconductor/metal ratio
        if cond:
            run_id += f"_cond_cls{cls_weight:.2f}_ratio{sc_metal_ratio:.2f}_thresh{metal_thresh:.2f}"
        
    # Add second task weight
        run_id += f"_task2w{task2_weight:.2f}"
        
    # Explicitly indicate auxiliary tasks status whether enabled or disabled
        run_id += f"_aux{'+'if aux else '-'}"
        
    # If auxiliary tasks enabled, add aux task weights
        if aux:
            run_id += f"_aux1w{aux1_weight:.2f}_aux2w{aux2_weight:.2f}_aux3w{aux3_weight:.2f}"
            
    # Add min bandgap, learning rate, epochs and weight decay identifiers
        run_id += f"_minbg{min_bandgap:.3f}_lr{lr_value}_e{epoch_value}_wd{float(wd_value)}"

        print(f"\n{'='*80}\nRunning combination: {run_id}\n{'='*80}")
        
        # Build command
        cmd = [
            "python", "main.py", data_dir,
            "--scheduler", sched,
            "--pooling", pool,
            "--num-splits", str(num_splits),
            "--epochs", str(epoch_value),
            "--lr", str(lr_value),
            "--weight-decay", str(wd_value),
            "--task2-weight", str(task2_weight),
            "--classification-weight", str(cls_weight),
            "--metal-weight", str(metal_weight),
            "--semiconductor-weight", str(semiconductor_weight),
            "--metal-threshold", str(metal_thresh),
            "--min-bandgap", str(min_bandgap)
        ]
        
    # Add various flag parameters
        if adaptive:
            cmd.append("--adaptive-weights")
            
    # Add conditional bandgap option
        if cond:
            cmd.append("--conditional-loss")
            
    # Add auxiliary tasks options
        if aux:
            cmd.append("--auxiliary-tasks")
            cmd.append("--aux-task1-weight")
            cmd.append(str(aux1_weight))  # set auxiliary task 1 weight (electronegativity variance)
            cmd.append("--aux-task2-weight")
            cmd.append(str(aux2_weight))  # set auxiliary task 2 weight (formation energy)
            cmd.append("--aux-task3-weight")
            cmd.append(str(aux3_weight))  # set auxiliary task 3 weight (first ionization energy mean)
        
        # Execute main.py with the parameters
        try:
            # Print the command to execute (useful for debugging)
            print("Executing:", " ".join(cmd))
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running combination {run_id}: {e}")
            continue  # continue to the next combination
        
        # After run completes, extract results
        metrics_filename = f"{run_id}_metrics.csv"
        metrics_path = result_dir / metrics_filename
        
        try:
            # Read the metrics file
            with open(metrics_path, 'r') as metrics_file:
                reader = csv.reader(metrics_file)
                header = next(reader)  # Get header row
                
                # Identify if this is a conditional loss run by checking the header
                is_conditional_run = len(header) > 4 and 'MaterialType_ACC' in header
                mt_acc_idx = header.index('MaterialType_ACC') if is_conditional_run else -1
                mt_loss_idx = header.index('MaterialType_Loss') if is_conditional_run else -1
                
                # Collect all run data
                run_metrics = []
                for row in reader:
                    if len(row) >= 4:  # Ensure we have the minimum expected columns
                        metrics_row = [float(row[0]), float(row[1]), float(row[2]), float(row[3])]
                        
                        # Add conditional loss metrics if available
                        if is_conditional_run and len(row) > mt_loss_idx:
                            metrics_row.append(float(row[mt_acc_idx]))  # Material Type ACC
                            metrics_row.append(float(row[mt_loss_idx]))  # Material Type Loss
                        else:
                            metrics_row.extend([None, None])  # No conditional loss metrics
                            
                        run_metrics.append(metrics_row)
                
                if run_metrics:
                    # Calculate averages for work function and band gap metrics
                    avg_wf_mae = sum(metric[0] for metric in run_metrics) / len(run_metrics)
                    avg_wf_r2 = sum(metric[1] for metric in run_metrics) / len(run_metrics)
                    avg_bg_mae = sum(metric[2] for metric in run_metrics) / len(run_metrics)
                    avg_bg_r2 = sum(metric[3] for metric in run_metrics) / len(run_metrics)
                    
                    # Calculate averages for material type metrics if available
                    avg_mt_acc = None
                    avg_mt_loss = None
                    if is_conditional_run:
                        # Filter out None values
                        valid_mt_metrics = [m for m in run_metrics if m[4] is not None]
                        if valid_mt_metrics:
                            avg_mt_acc = sum(metric[4] for metric in valid_mt_metrics) / len(valid_mt_metrics)
                            avg_mt_loss = sum(metric[5] for metric in valid_mt_metrics) / len(valid_mt_metrics)
                      # Update summary file
                    with open(summary_all_path, 'a', newline='') as summary_file:
                        writer = csv.writer(summary_file)
                        writer.writerow([
                            sched,
                            pool,
                            'Yes' if adaptive else 'No',
                            'Yes' if cond else 'No',    # Conditional loss setting
                            f'{cls_weight}',            # Classification weight
                            f'{sc_metal_ratio}',        # Semiconductor/metal weight ratio
                            f'{metal_thresh}',          # Metal classification threshold
                            f'{metal_weight:.4f}',      # Metal weight
                            f'{semiconductor_weight:.4f}', # Semiconductor weight
                            f'{task2_weight}',          # Second task weight
                            'Yes' if aux else 'No',     # Auxiliary tasks setting
                            f'{aux1_weight}' if aux else 'N/A',  # Aux task1 weight
                            f'{aux2_weight}' if aux else 'N/A',  # Aux task2 weight
                            f'{aux3_weight}' if aux else 'N/A',  # Aux task3 weight
                            f'{min_bandgap}',           # Minimum bandgap threshold
                            f'{lr_value}',              # Learning rate
                            f'{epoch_value}',           # Epochs
                            f'{wd_value}',              # Weight decay
                            f'{avg_wf_mae:.6f}',
                            f'{avg_wf_r2:.6f}',
                            f'{avg_bg_mae:.6f}',
                            f'{avg_bg_r2:.6f}',
                            f'{avg_mt_acc:.6f}' if avg_mt_acc is not None else 'N/A',
                            f'{avg_mt_loss:.6f}' if avg_mt_loss is not None else 'N/A'
                        ])
                    
                    print(f"Results for {run_id}:")
                    print(f"  Learning Rate: {lr_value}, Epochs: {epoch_value}, Weight Decay: {wd_value}")
                    print(f"  Classification Weight: {cls_weight}, SC-Metal Ratio: {sc_metal_ratio}, Metal Threshold: {metal_thresh}")
                    print(f"  Metal Weight: {metal_weight:.4f}, SC Weight: {semiconductor_weight:.4f}, Task2 Weight: {task2_weight}")
                    print(f"  Min Bandgap: {min_bandgap}")
                    if aux:
                        print(f"  Auxiliary Task Weights: Aux1={aux1_weight}, Aux2={aux2_weight}, Aux3={aux3_weight}")
                    print(f"  Work Function: MAE = {avg_wf_mae:.6f}, R² = {avg_wf_r2:.6f}")
                    print(f"  Band Gap: MAE = {avg_bg_mae:.6f}, R² = {avg_bg_r2:.6f}")
                    if avg_mt_acc is not None:
                        print(f"  Material Type: ACC = {avg_mt_acc:.6f}, Loss = {avg_mt_loss:.6f}")
                        
        except Exception as e:
            print(f"Error processing results for {run_id}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nGrid search complete!")
    print("Results are saved in the 'result' directory")
    print(f"Summary of all combinations is available in '{summary_all_path}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run grid search for CGCNN Multi-Task Learning')
    parser.add_argument('data_dir', help='Path to the data directory')
    parser.add_argument('--num-splits', type=int, default=3,
                        help='Number of runs per parameter combination (default: 3)')
    parser.add_argument('--epochs', type=int, default=80,
                        help='Number of epochs for training (default: 80)')
    
    args = parser.parse_args()
    run_grid_search(args.data_dir, args.num_splits, args.epochs)
