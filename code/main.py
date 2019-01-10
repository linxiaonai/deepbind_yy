import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from preprocessing import ChipToMatrix
from datasets import DeepMotifDataset
from models import CNN
from settings import file_name, batch_size, num_workers, \
                     model_type, dropout_prob_lst, \
                     optim, milestones, lr_decay, learning_rate_lst, weight_decay_lst, \
                     max_epoch, max_not_improvement, grad_clip
from utils import cal_acc_auc, save_best, clip_gradient, load_best_model, to_csv


device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32
loss_func = nn.MSELoss()
best_auc_all = 0                    # for all lr and weight decay

def train(loader_train, loader_val, learning_rate, weight_decay, dropout_pro, train_num):
    global best_auc_all
    epochs_since_improvement = 0
    best_auc_this = 0
    last_auc = 0

    if model_type == 'CNN':
        model = CNN(dropout_pro=dropout_pro)
    model = model.to(device=device)
    if device == 'cuda':
        model = nn.DataParallel(model, device_ids=[0, 1])

    print("\n\nModel type: {} \nModel: {} ".format(model_type, model))
    print("\nLearning_rate:{}\t weight_decay:{}\t dropout_prob:{}".format(learning_rate, weight_decay, dropout_pro))
    print("Training...............................................")
    if optim == 'SGD':
        optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif optim == 'Adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate,
                                     betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=lr_decay)

    for epoch in range(max_epoch):
        scheduler.step()
        # early stopping, terminate training after there is no or little improvement for 20 epochs
        if epochs_since_improvement == max_not_improvement:
            break

        train_one_epoch(loader_train, model, optimizer, dtype, train_num)
        train_acc, train_auc = cal_acc_auc(loader_train, model, device, dtype)
        val_acc, val_auc = cal_acc_auc(loader_val, model, device, dtype)
        print('Epoch:{} \t train acc:{:.4f} \t train auc: {:.4f}'
              ' validation acc: {:.4f} \t validation auc: {:.4f}'
              .format(epoch, train_acc, train_auc, val_acc, val_auc))


        is_best_this = val_auc > best_auc_this
        is_best_all = val_auc > best_auc_all
        if not is_best_this:
            epochs_since_improvement += 1
        else:
            best_auc_this = val_auc
            epochs_since_improvement = 0

        if is_best_all:
            best_auc_all = val_auc
            save_best(model, model_type, file_name, PATH='../model/',)
        print("Best val_auc for all models: {:.6f} \t Best val_auc for this model: {:.6f}"
              .format(best_auc_all, best_auc_this))

        # break if not converge
        if np.linalg.norm(last_auc - val_auc) < 1e-5 and epoch > 3:
            break
        last_auc = val_auc


def train_one_epoch(loader_train, model, optimizer, dtype, train_num):
    step_per_epoch = math.ceil(train_num // batch_size)
    t = tqdm(total=step_per_epoch)
    for step, (X, y) in enumerate(loader_train):
        t.update(1)
        model.train()
        X = X.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=torch.long)

        scores = model(X)
        prob = F.softmax(scores, dim=1)
        _, predict_labels = scores.max(dim=1)
        target = np.zeros([y.size(0), 2])
        for i in range(len(y)):
            if (y[i] == 0):
                target[i] = [1, 0]
            else:
                target[i] = [0, 1]
        target = torch.from_numpy(target).to(device=device, dtype=dtype)
        loss = loss_func(prob, target)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            clip_gradient(optimizer, grad_clip)

        # Update weight
        optimizer.step()

        # # Print loss adn train acc and auc during training
        # y = y.to(device='cpu', dtype=torch.long)
        # predict_labels = predict_labels.to(device='cpu', dtype=torch.long)
        # train_acc = metrics.accuracy_score(y, predict_labels)
        # train_auc = metrics.roc_auc_score(y, predict_labels)
        # if step % loss_print == 0:
        #     print("Step: {}, loss: {:.8f}, train_acc: {:.4f}, train_auc: {:.4f}"
        #           .format(step, loss.item(), train_acc, train_auc))
    t.close()


def eval(model_type, loader, device, dtype, file_name):
    best_model = load_best_model(model_type, file_name, PATH='../model/')
    best_model.eval()
    acc, auc = cal_acc_auc(loader, best_model, device, dtype)
    return acc, auc


def main():
    print("\nUsing device:", device)

    print("\nPreprocessing the dataset..............................")
    train_dataset, val_dataset = ChipToMatrix(file_name, 'train')
    test_dataset = ChipToMatrix(file_name, 'test')
    train_num = len(train_dataset[1])

    loader_train = torch.utils.data.DataLoader(DeepMotifDataset(dataset=train_dataset),
                                               batch_size=batch_size, shuffle=True, num_workers=num_workers)
    loader_val = torch.utils.data.DataLoader(DeepMotifDataset(dataset=val_dataset),
                                             batch_size=batch_size, shuffle=True, num_workers=num_workers)
    loader_test = torch.utils.data.DataLoader(DeepMotifDataset(dataset=test_dataset),
                                              batch_size=batch_size, shuffle=True, num_workers=num_workers)

    for learning_rate in learning_rate_lst:
        for weight_decay in weight_decay_lst:
            for dropout_prob in dropout_prob_lst:
                train(loader_train, loader_val, learning_rate, weight_decay, dropout_prob, train_num)

    # Evaluation this model on test dataset
    test_acc, test_auc = eval(model_type, loader_test, device, dtype, file_name)
    print("The best Model {} for file {} \n  test acc: {:.4f} \t test auc: {:.4f}"
          .format(model_type, file_name, test_acc, test_auc))
    to_csv(file_name, model_type, test_acc, test_auc)


if __name__ == "__main__":
    main()