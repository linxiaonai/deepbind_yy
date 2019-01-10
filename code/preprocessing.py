# Proprocess the data by One-hot encoding, referring to DeepMotif

import pandas as pd
from settings import bases

def OneHotEncoding(seq):
    """
    :param seq: 'ATCCGGTT' like
    :return: a [4, 101] matrix
    """
    dummies = pd.get_dummies(list(seq))
    onehot_encoded = dummies.T.reindex(bases).fillna(0).values
    return onehot_encoded

def ChipToMatrix(file_name, file_type):
    """
    :param file_name: ATF1_K562_ATF1_-06-325-_Harvard
    :param file_type: train or test
    :return: train_dataset [train_seqs (n, 4, 101), train_labels (n, )]
             val_dataset   [test_seqs (n, 4, 101), test_labels (n, )]
    """
    file = '../data/deepbind/' + str(file_name) + '/' + \
           str(file_type) + '.fa'
    seqs = []
    labels = []
    with open(file, 'r') as f:
        for line in f:
            if line[0] in 'ACTGU':
                seq = line[:-1]
                seqs.append(OneHotEncoding(seq))
            if line[0] == '>':
                labels.append(int(line[-2]))

    if file_type == 'train':
        num_val = 1000
        train_dataset = [seqs[:-num_val], labels[:-num_val]]
        val_dataset = [seqs[-num_val:], labels[-num_val:]]
        return train_dataset, val_dataset
    if file_type == 'test':
        test_dataset = [seqs, labels]
        return test_dataset

# test
# train_dataset, val_dataset = ChipToMatrix('ATF1_K562_ATF1_-06-325-_Harvard', 'train')


