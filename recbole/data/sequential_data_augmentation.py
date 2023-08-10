import torch
import random


class BlankProlongAugmentation:
    def __init__(self, config, dataset):
        self.fix = config['fix_prolong_length'] if config['fix_prolong_length'] is not None else False
        self.max_length = config['max_sequence_length'] if config['max_sequence_length'] is not None else 50
        self.max_shifts = config['max_prolong_shifts'] if config['max_prolong_shifts'] is not None else 20
        self.dataset = dataset

    def get_augmentation_data(self, item_seq, item_seq_len):
        item_seq = item_seq.clone().detach()
        item_seq_len = item_seq_len.clone().detach()
        for i in range(len(item_seq)):
            d = min(self.max_shifts, self.max_length - item_seq_len[i])
            d = d if self.fix else torch.randint(low=1, high=d, size=(1,)).item()
            item_seq[i] = item_seq[i].roll(shifts=d)
            item_seq_len[i] += d
        return item_seq, item_seq_len


class BlankInsertionAugmentation:
    def __init__(self, config, dataset):
        self.max_length = config['max_sequence_length'] if config['max_sequence_length'] is not None else 50
        self.dataset = dataset

    def get_augmentation_data(self, item_seq, item_seq_len):
        item_seq = item_seq.clone().detach()
        item_seq_len = item_seq_len.clone().detach()
        for i in range(len(item_seq)):
            d = min(item_seq.shape[1], self.max_length)
            id_list = torch.LongTensor(random.sample(range(0, d), item_seq_len[i])).sort()[0]
            item_list = item_seq[i, 0: item_seq_len[i]]
            item_seq[i] *= 0
            item_seq[i, id_list] = item_list.clone().detach()
            item_seq_len[i] = id_list[-1] + 1
        item_seq = item_seq.contiguous().cuda()
        item_seq_len = item_seq_len.contiguous().cuda()
        return item_seq, item_seq_len

