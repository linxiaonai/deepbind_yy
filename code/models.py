# Class CNN, RNN and CNN-RNN

import torch.nn as nn

from settings import wordvec_dim, cnn_size, cnn_filters, cnn_poolsize, hidden_layer_size, num_classes

class CNN(nn.Module):
    """
    CNN
    """
    def __init__(self, dropout_pro=0.2):
        super().__init__()
        in_channel = wordvec_dim
        self.conv = nn.Sequential()
        self.conv.add_module(name='conv1', module=nn.Conv1d(in_channel, cnn_size, kernel_size=cnn_filters[0]))
        self.conv.add_module(name='relu1', module=nn.ReLU())
        self.conv.add_module(name='maxpool1', module=nn.MaxPool1d(kernel_size=cnn_poolsize))
        self.conv.add_module(name='dropout1', module=nn.Dropout(p=dropout_pro))

        self.conv.add_module(name='conv2', module=nn.Conv1d(cnn_size, cnn_size, kernel_size=cnn_filters[1]))
        self.conv.add_module(name='relu2', module=nn.ReLU())
        self.conv.add_module(name='maxpool2', module=nn.MaxPool1d(kernel_size=cnn_poolsize))
        self.conv.add_module(name='dropout2', module=nn.Dropout(p=dropout_pro))

        self.conv.add_module(name='conv3', module=nn.Conv1d(cnn_size, cnn_size, kernel_size=cnn_filters[2]))
        self.conv.add_module(name='relu3', module=nn.ReLU())
        self.conv.add_module(name='dropout3', module=nn.Dropout(p=dropout_pro))

        self.dense = nn.Sequential()
        self.dense.add_module(name='fc1', module=nn.Linear(128 * 19, hidden_layer_size))        # (batch_size, hidden_layer_size)
        self.dense.add_module(name='ReLU', module=nn.ReLU())
        self.dense.add_module(name='fc2', module=nn.Linear(hidden_layer_size, num_classes))     # (batch_size, num_classes)

    def forward(self, x):
        conv3_out = self.conv(x)
        conv_out = conv3_out.view(conv3_out.size(0), -1)
        scores = self.dense(conv_out)
        return scores


