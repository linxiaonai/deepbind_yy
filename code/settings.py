

# Global settings
file_name = 'CEBPB_K562_CEBPB_-SC-150-_HudsonAlpha'
bases = ['A', 'C', 'T', 'G']
batch_size = 256                # the number of training example in each DataLoader
num_workers = 1                 # for data loading

# Parameters for models
# CNN(cnn_size=128, cnn_filters=[9, 5, 3], cnn_poolsize=2, dropout_pro=0.5, hidden_layer_size=128, num_classes=2)
model_type = 'CNN'              # 'CNN' or 'RNN' or 'CNN-RNN'
wordvec_dim = 4                 # 4 for 'ATCG'
cnn_size = 128                  # out_channels after the first conv1d
cnn_filters = [9, 5, 3]         # convolution layer filter sizes
cnn_poolsize = 2                # maxpooling kernel size
dropout_prob_lst = [0.2, 0.5]   # probability of an element to be zeroed
hidden_layer_size = 128         # linear hidden layer size
num_classes = 2

# Optimizer
optim = 'Adam'                  # 'SGD' or 'Adam'
milestones = [20, 30, 40]
lr_decay = 0.5

# Parameters during training
learning_rate_lst = [1e-2, 1e-3]
weight_decay_lst = [0, 1e-5]
max_epoch = 1
max_not_improvement = 20

# loss_print = 1
grad_clip = 5                   # gradient clip value magnitude, 0 if there is no gradient clip

