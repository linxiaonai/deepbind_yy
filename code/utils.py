import torch
import os

import numpy as np
import pandas as pd
from sklearn import metrics

def cal_acc_auc(loader, model, device, dtype):
    with torch.no_grad():
        acc_lst = []
        auc_lst = []
        for X, y in loader:
            X = X.to(device=device, dtype=dtype)
            y = y.to(device='cpu', dtype=torch.long)
            scores = model(X)
            _, predict_labels = scores.max(dim=1)
            predict_labels = predict_labels.to(device='cpu', dtype=torch.long)
            acc_lst.append(metrics.accuracy_score(y, predict_labels))
            auc_lst.append(metrics.roc_auc_score(y, predict_labels))
        acc = np.mean(acc_lst)
        auc = np.mean(auc_lst)
    return acc, auc


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_best(model, model_type, file_name, PATH='../model/'):
    if not os.path.exists(PATH):
        os.mkdir(PATH)
    if not os.path.exists(PATH + file_name + '/'):
        os.mkdir(PATH + file_name + '/')
    # for future training
    # state = {'eooch': epoch,
    #          'epochs_since_improvement': epochs_since_improvement,
    #          'model': model.state_dict(),
    #          'optimizer': optimizer.state_dict(),
    #          'val_acc': val_acc,
    #          'val_auc': val_auc}
    # filename = model_type + '_checkpoint_epoch_' + str(epoch) + '.pt'
    # torch.save(state, PATH + filename)
    model_name = PATH + file_name + '/' + 'Best_' + model_type + '.pt'
    torch.save(model, model_name)


def load_best_model(model_type, file_name, PATH='../model/'):
    print("\nInitializing the best {} model..................".format(model_type))
    return torch.load(PATH + file_name + '/' + 'Best_' + model_type + '.pt')
    # torch.load('../model/CNN_checkpoint_epoch_0.pth.tar')


def to_csv(file_name, model_type, test_acc, test_auc,
           result_csv_name='scores_all_models.csv', PATH_CSV='../results/'):
    if not os.path.exists(PATH_CSV):
        os.mkdir(PATH_CSV)
    print('Save results into ', PATH_CSV)
    df = pd.DataFrame({'file_name': [file_name], 'model_type': [model_type],
                       'test_acc': [test_acc], 'test_auc': [test_auc]})
    path_file = PATH_CSV + result_csv_name
    if os.path.isfile(path_file):
        old_df = pd.read_csv(path_file)
        new_df = old_df[old_df['file_name'] != file_name]
        new_df = new_df.append(df, ignore_index=True)
        new_df.to_csv(path_file, header=['file_name', 'model_type', 'test_acc', 'test_auc'], index=False)
    else:
        df.to_csv(path_file, index=False)