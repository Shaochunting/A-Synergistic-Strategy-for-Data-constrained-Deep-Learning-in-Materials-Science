import argparse
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable
from torch.utils.data import DataLoader

from cgcnn.data import CIFData
from cgcnn.data import collate_pool
from cgcnn.model import CrystalGraphConvNet

parser = argparse.ArgumentParser(description='Crystal gated neural networks')
parser.add_argument('modelpath', help='path to the trained model.')
parser.add_argument('cifpath', help='path to the directory of CIF files.')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

args = parser.parse_args(sys.argv[1:])
if os.path.isfile(args.modelpath):
    print("=> loading model params '{}'".format(args.modelpath))
    model_checkpoint = torch.load(args.modelpath,
                                  map_location=lambda storage, loc: storage)
    model_args = argparse.Namespace(**model_checkpoint['args'])
    print("=> loaded model params '{}'".format(args.modelpath))
else:
    print("=> no model params found at '{}'".format(args.modelpath))

args.cuda = not args.disable_cuda and torch.cuda.is_available()

if model_args.task == 'regression':
    best_mae_error = 1e10
else:
    best_mae_error = 0.


def main():
    global args, model_args, best_mae_error

    # load data
    dataset = CIFData(args.cifpath)
    collate_fn = collate_pool
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.workers, collate_fn=collate_fn,
                             pin_memory=args.cuda)

    # build model
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    
    model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                atom_fea_len=model_args.atom_fea_len,
                                n_conv=model_args.n_conv,
                                h_fea_len=model_args.h_fea_len,
                                n_h=model_args.n_h,
                                classification=True if model_args.task ==
                                'classification' else False,
                                pooling_method=getattr(model_args, 'pooling', 'mean'))
    if args.cuda:
        model.cuda()

    # define loss func and optimizer
    if model_args.task == 'classification':
        criterion = nn.NLLLoss()
    else:
        criterion = nn.MSELoss()
    # if args.optim == 'SGD':
    #     optimizer = optim.SGD(model.parameters(), args.lr,
    #                           momentum=args.momentum,
    #                           weight_decay=args.weight_decay)
    # elif args.optim == 'Adam':
    #     optimizer = optim.Adam(model.parameters(), args.lr,
    #                            weight_decay=args.weight_decay)
    # else:
    #     raise NameError('Only SGD or Adam is allowed as --optim')

    normalizer_task1 = Normalizer(torch.zeros(1))
    normalizer_task2 = Normalizer(torch.zeros(1))

    # optionally resume from a checkpoint
    if os.path.isfile(args.modelpath):
        print("=> loading model '{}'".format(args.modelpath))
        checkpoint = torch.load(args.modelpath,
                                map_location=lambda storage, loc: storage)
        
        # Load model state dict, but ignore auxiliary task layers if they exist
        model_state_dict = checkpoint['state_dict']
        
        # Filter out auxiliary task layers from the saved state dict
        filtered_state_dict = {}
        for key, value in model_state_dict.items():
            if not any(aux_key in key for aux_key in ['fc_out_aux_task1', 'fc_out_aux_task2', 'fc_out_aux_task3']):
                filtered_state_dict[key] = value
        
        # Load the filtered state dict
        model.load_state_dict(filtered_state_dict, strict=False)
        
        normalizer_task1.load_state_dict(checkpoint['normalizer_task1'])
        normalizer_task2.load_state_dict(checkpoint['normalizer_task2'])
        print("=> loaded model '{}' (epoch {}, validation {})"
              .format(args.modelpath, checkpoint['epoch'],
                      checkpoint['best_mae_error']))
    else:
        print("=> no model found at '{}'".format(args.modelpath))

    validate(test_loader, model, criterion, normalizer_task1, normalizer_task2, test=True)


def validate(val_loader, model, criterion, normalizer_task1, normalizer_task2, test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    if model_args.task == 'regression':
        mae_errors_task1 = AverageMeter()  # Work function
        mae_errors_task2 = AverageMeter()  # Band gap
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

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        with torch.no_grad():
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
        
        if model_args.task == 'regression':
            # For multi-task regression, target has shape [batch_size, 2] if targets exist
            if target is not None and target.numel() > 0:
                target_task1 = target[:, 0].view(-1, 1)  # Work function
                target_task2 = target[:, 1].view(-1, 1)  # Band gap
                target_normed_task1 = normalizer_task1.norm(target_task1)
                target_normed_task2 = normalizer_task2.norm(target_task2)
            else:
                # No targets available for prediction-only mode
                target_task1 = torch.zeros(input[0].size(0), 1)
                target_task2 = torch.zeros(input[0].size(0), 1)
                target_normed_task1 = torch.zeros(input[0].size(0), 1)
                target_normed_task2 = torch.zeros(input[0].size(0), 1)
        else:
            target_normed = target.view(-1).long() if target is not None else torch.zeros(input[0].size(0)).long()
            
        with torch.no_grad():
            if args.cuda:
                if model_args.task == 'regression':
                    target_var_task1 = Variable(target_normed_task1.cuda(non_blocking=True))
                    target_var_task2 = Variable(target_normed_task2.cuda(non_blocking=True))
                else:
                    target_var = Variable(target_normed.cuda(non_blocking=True))
            else:
                if model_args.task == 'regression':
                    target_var_task1 = Variable(target_normed_task1)
                    target_var_task2 = Variable(target_normed_task2)
                else:
                    target_var = Variable(target_normed)

        # compute output - for multi-task model
        if model_args.task == 'regression':
            # Multi-task model returns 5 outputs: pred_task1, pred_task2, material_type_probs, atom_features_list, aux_outputs
            # We don't use auxiliary tasks during prediction, so set use_auxiliary_tasks=False
            output_task1, output_task2, output_material_type, atom_features_list, aux_outputs = model(*input_var, use_auxiliary_tasks=False)
            
            # Calculate losses only if we have real targets
            if target is not None and target.numel() > 0:
                loss_task1 = criterion(output_task1, target_var_task1)
                loss_task2 = criterion(output_task2, target_var_task2)
                loss = 0.5 * loss_task1 + 0.5 * loss_task2  # Simple equal weighting
                
                # Calculate MAE for both tasks
                mae_error_task1 = mae(normalizer_task1.denorm(output_task1.data.cpu()), target_task1)
                mae_error_task2 = mae(normalizer_task2.denorm(output_task2.data.cpu()), target_task2)
                
                losses.update(loss.data.cpu().item(), target_task1.size(0))
                mae_errors_task1.update(mae_error_task1, target_task1.size(0))
                mae_errors_task2.update(mae_error_task2, target_task2.size(0))
            else:
                # Prediction-only mode - no loss calculation
                loss = torch.tensor(0.0)
                mae_error_task1 = torch.tensor(0.0)
                mae_error_task2 = torch.tensor(0.0)
            
            if test:
                test_pred_task1 = normalizer_task1.denorm(output_task1.data.cpu())
                test_pred_task2 = normalizer_task2.denorm(output_task2.data.cpu())
                
                test_preds_task1 += test_pred_task1.view(-1).tolist()
                test_preds_task2 += test_pred_task2.view(-1).tolist()
                
                if target is not None and target.numel() > 0:
                    test_targets_task1 += target_task1.view(-1).tolist()
                    test_targets_task2 += target_task2.view(-1).tolist()
                else:
                    # No targets - just add placeholder values
                    test_targets_task1 += [0.0] * test_pred_task1.size(0)
                    test_targets_task2 += [0.0] * test_pred_task2.size(0)
                
                test_cif_ids += batch_cif_ids
        else:
            output = model(*input_var)
            loss = criterion(output, target_var)
            accuracy, precision, recall, fscore, auc_score =\
                class_eval(output.data.cpu(), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))
            if test:
                test_pred = torch.exp(output.data.cpu())
                test_target = target
                assert test_pred.shape[1] == 2
                test_preds_task1 += test_pred[:, 1].tolist()
                test_targets_task1 += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if model_args.task == 'regression':
                if target is not None and target.numel() > 0:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'WF-MAE {mae1.val:.3f} ({mae1.avg:.3f})\t'
                          'BG-MAE {mae2.val:.3f} ({mae2.avg:.3f})'.format(
                           i, len(val_loader), batch_time=batch_time, loss=losses,
                           mae1=mae_errors_task1, mae2=mae_errors_task2))
                else:
                    print('Predict: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Predicting work function and band gap...'.format(
                           i, len(val_loader), batch_time=batch_time))
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
        with open('predictions_results.csv', 'w') as f:
            writer = csv.writer(f)
            if model_args.task == 'regression':
                # Check if we have real targets or just predictions
                has_targets = any(t != 0.0 for t in test_targets_task1[:5])  # Check first few entries
                
                if has_targets:
                    writer.writerow(['cif_id', 'work_function_target', 'work_function_pred', 
                                    'band_gap_target', 'band_gap_pred'])
                    for cif_id, target1, pred1, target2, pred2 in zip(test_cif_ids, 
                                                                      test_targets_task1, test_preds_task1,
                                                                      test_targets_task2, test_preds_task2):
                        writer.writerow((cif_id, target1, pred1, target2, pred2))
                else:
                    # Prediction-only mode
                    writer.writerow(['cif_id', 'work_function_pred', 'band_gap_pred'])
                    for cif_id, pred1, pred2 in zip(test_cif_ids, test_preds_task1, test_preds_task2):
                        writer.writerow((cif_id, pred1, pred2))
            else:
                writer.writerow(['cif_id', 'target', 'pred'])
                for cif_id, target, pred in zip(test_cif_ids, test_targets_task1,
                                                test_preds_task1):
                    writer.writerow((cif_id, target, pred))
        
        print(f"Predictions saved to: predictions_results.csv")
    else:
        star_label = '*'
        
    if model_args.task == 'regression':
        if target is not None and target.numel() > 0:
            print(' {star} WF-MAE {mae1.avg:.3f} BG-MAE {mae2.avg:.3f}'.format(
                  star=star_label, mae1=mae_errors_task1, mae2=mae_errors_task2))
            return mae_errors_task1.avg, mae_errors_task2.avg
        else:
            print(' {star} Prediction completed for {n} materials'.format(
                  star=star_label, n=len(test_cif_ids)))
            return 0.0, 0.0
    else:
        print(' {star} AUC {auc.avg:.3f}'.format(star=star_label,
                                                 auc=auc_scores))
        return auc_scores.avg


class Normalizer(object):
    """Normalize a Tensor and restore it later. """
    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

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


if __name__ == '__main__':
    main()