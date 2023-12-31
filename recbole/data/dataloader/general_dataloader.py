# @Time   : 2020/7/7
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time   : 2020/9/9, 2020/9/29, 2021/7/15
# @Author : Yupeng Hou, Yushuo Chen, Xingyu Pan
# @email  : houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn, xy_pan@foxmail.com

"""
recbole.data.dataloader.general_dataloader
################################################
"""

import numpy as np
import torch
import random
import math

from recbole.data.dataloader.abstract_dataloader import AbstractDataLoader, NegSampleDataLoader
from recbole.data.interaction import Interaction, cat_interactions
from recbole.utils import InputType, ModelType


class TrainDataLoader(NegSampleDataLoader):
    """:class:`TrainDataLoader` is a dataloader for training.
    It can generate negative interaction when :attr:`training_neg_sample_num` is not zero.
    For the result of every batch, we permit that every positive interaction and its negative interaction
    must be in the same batch.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self._set_neg_sample_args(config, dataset, config['MODEL_INPUT_TYPE'], config['train_neg_sample_args'])
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config['train_batch_size']
        if self.neg_sample_args['strategy'] == 'by':
            batch_num = max(batch_size // self.times, 1)
            new_batch_size = batch_num * self.times
            self.step = batch_num
            self.set_batch_size(new_batch_size)
        else:
            self.step = batch_size
            self.set_batch_size(batch_size)

    def update_config(self, config):
        self._set_neg_sample_args(config, self.dataset, config['MODEL_INPUT_TYPE'], config['train_neg_sample_args'])
        super().update_config(config)

    @property
    def pr_end(self):
        return len(self.dataset)

    def _shuffle(self):
        self.dataset.shuffle()

    def _next_batch_data(self):
        cur_data = self._neg_sampling(self.dataset[self.pr:self.pr + self.step])
        if self.config['aug'] == "cts":
            self.data_aug(cur_data, slice(self.pr, self.pr + self.step))
        elif self.config['aug'] == 'cl4rec':
            self.cl4srec_aug(cur_data)
        self.pr += self.step
        return cur_data

    def data_aug(self, cur_data, index):
        cur_same_target = self.dataset.data_aug_index[index]
        null_index = []
        positives = []
        for i, targets in enumerate(cur_same_target):
            if len(targets) == 0:
                positives.append(-1)
                null_index.append(i)
            else:
                positives.append(np.random.choice(targets))

        temp_id = self.dataset.inter_feat['item_id_list'].detach().clone()
        temp_len = self.dataset.inter_feat['item_length'].detach().clone()

        aug_pos_seqs = temp_id[positives]
        aug_pos_lengths = temp_len[positives]
        if null_index:
            aug_pos_seqs[null_index] = cur_data['item_id_list'][null_index]
            aug_pos_lengths[null_index] = cur_data['item_length'][null_index]
        
        cur_data.update(Interaction({'aug': aug_pos_seqs, 'aug_lengths': aug_pos_lengths}))

    def cl4srec_aug(self, cur_data):
        def item_crop(seq, length, eta=0.6):
            num_left = math.floor(length * eta)
            crop_begin = random.randint(0, length - num_left)
            croped_item_seq = np.zeros(seq.shape[0])
            if crop_begin + num_left < seq.shape[0]:
                croped_item_seq[:num_left] = seq[crop_begin:crop_begin + num_left]
            else:
                croped_item_seq[:num_left] = seq[crop_begin:]
            return torch.tensor(croped_item_seq, dtype=torch.long), torch.tensor(num_left, dtype=torch.long)

        def item_mask(seq, length, gamma=0.3):
            num_mask = math.floor(length * gamma)
            mask_index = random.sample(range(length), k=num_mask)
            masked_item_seq = seq[:]
            masked_item_seq[mask_index] = self.dataset.item_num  # token 0 has been used for semantic masking
            return masked_item_seq, length

        def item_reorder(seq, length, beta=0.6):
            num_reorder = math.floor(length * beta)
            reorder_begin = random.randint(0, length - num_reorder)
            reordered_item_seq = seq[:]
            shuffle_index = list(range(reorder_begin, reorder_begin + num_reorder))
            random.shuffle(shuffle_index)
            reordered_item_seq[reorder_begin:reorder_begin + num_reorder] = reordered_item_seq[shuffle_index]
            return reordered_item_seq, length

        seqs = cur_data['item_id_list'].detach().clone()
        lengths = cur_data['item_length'].detach().clone()

        aug_seq1 = []
        aug_len1 = []
        aug_seq2 = []
        aug_len2 = []
        for seq, length in zip(seqs, lengths):
            if length > 1:
                switch = random.sample(range(3), k=2)
            else:
                switch = [3, 3]
                aug_seq = seq
                aug_len = length
            if switch[0] == 0:
                aug_seq, aug_len = item_crop(seq, length)
            elif switch[0] == 1:
                aug_seq, aug_len = item_mask(seq, length)
            elif switch[0] == 2:
                aug_seq, aug_len = item_reorder(seq, length)

            aug_seq1.append(aug_seq)
            aug_len1.append(aug_len)

            if switch[1] == 0:
                aug_seq, aug_len = item_crop(seq, length)
            elif switch[1] == 1:
                aug_seq, aug_len = item_mask(seq, length)
            elif switch[1] == 2:
                aug_seq, aug_len = item_reorder(seq, length)

            aug_seq2.append(aug_seq)
            aug_len2.append(aug_len)

        cur_data.update(Interaction({'aug1': torch.stack(aug_seq1), 'aug_len1': torch.stack(aug_len1),
                                     'aug2': torch.stack(aug_seq2), 'aug_len2': torch.stack(aug_len2)}))


class NegSampleEvalDataLoader(NegSampleDataLoader):
    """:class:`NegSampleEvalDataLoader` is a dataloader for neg-sampling evaluation.
    It is similar to :class:`TrainDataLoader` which can generate negative items,
    and this dataloader also permits that all the interactions corresponding to each user are in the same batch
    and positive interactions are before negative interactions.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self._set_neg_sample_args(config, dataset, InputType.POINTWISE, config['eval_neg_sample_args'])
        if self.neg_sample_args['strategy'] == 'by':
            user_num = dataset.user_num
            dataset.sort(by=dataset.uid_field, ascending=True)
            self.uid_list = []
            start, end = dict(), dict()
            for i, uid in enumerate(dataset.inter_feat[dataset.uid_field].numpy()):
                if uid not in start:
                    self.uid_list.append(uid)
                    start[uid] = i
                end[uid] = i
            self.uid2index = np.array([None] * user_num)
            self.uid2items_num = np.zeros(user_num, dtype=np.int64)
            for uid in self.uid_list:
                self.uid2index[uid] = slice(start[uid], end[uid] + 1)
                self.uid2items_num[uid] = end[uid] - start[uid] + 1
            self.uid_list = np.array(self.uid_list)

        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config['eval_batch_size']
        if self.neg_sample_args['strategy'] == 'by':
            inters_num = sorted(self.uid2items_num * self.times, reverse=True)
            batch_num = 1
            new_batch_size = inters_num[0]
            for i in range(1, len(inters_num)):
                if new_batch_size + inters_num[i] > batch_size:
                    break
                batch_num = i + 1
                new_batch_size += inters_num[i]
            self.step = batch_num
            self.set_batch_size(new_batch_size)
        else:
            self.step = batch_size
            self.set_batch_size(batch_size)

    def update_config(self, config):
        self._set_neg_sample_args(config, self.dataset, InputType.POINTWISE, config['eval_neg_sample_args'])
        super().update_config(config)

    @property
    def pr_end(self):
        if self.neg_sample_args['strategy'] == 'by':
            return len(self.uid_list)
        else:
            return len(self.dataset)

    def _shuffle(self):
        self.logger.warnning('NegSampleEvalDataLoader can\'t shuffle')

    def _next_batch_data(self):
        if self.neg_sample_args['strategy'] == 'by':
            uid_list = self.uid_list[self.pr:self.pr + self.step]
            data_list = []
            idx_list = []
            positive_u = []
            positive_i = torch.tensor([], dtype=torch.int64)

            for idx, uid in enumerate(uid_list):
                index = self.uid2index[uid]
                data_list.append(self._neg_sampling(self.dataset[index]))
                idx_list += [idx for i in range(self.uid2items_num[uid] * self.times)]
                positive_u += [idx for i in range(self.uid2items_num[uid])]
                positive_i = torch.cat((positive_i, self.dataset[index][self.iid_field]), 0)

            cur_data = cat_interactions(data_list)
            idx_list = torch.from_numpy(np.array(idx_list))
            positive_u = torch.from_numpy(np.array(positive_u))

            self.pr += self.step

            return cur_data, idx_list, positive_u, positive_i
        else:
            cur_data = self._neg_sampling(self.dataset[self.pr:self.pr + self.step])
            self.pr += self.step
            return cur_data, None, None, None


class FullSortEvalDataLoader(AbstractDataLoader):
    """:class:`FullSortEvalDataLoader` is a dataloader for full-sort evaluation. In order to speed up calculation,
    this dataloader would only return then user part of interactions, positive items and used items.
    It would not return negative items.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.uid_field = dataset.uid_field
        self.iid_field = dataset.iid_field
        self.is_sequential = config['MODEL_TYPE'] == ModelType.SEQUENTIAL
        if not self.is_sequential:
            user_num = dataset.user_num
            self.uid_list = []
            self.uid2items_num = np.zeros(user_num, dtype=np.int64)
            self.uid2positive_item = np.array([None] * user_num)
            self.uid2history_item = np.array([None] * user_num)

            dataset.sort(by=self.uid_field, ascending=True)
            last_uid = None
            positive_item = set()
            uid2used_item = sampler.used_ids
            for uid, iid in zip(dataset.inter_feat[self.uid_field].numpy(), dataset.inter_feat[self.iid_field].numpy()):
                if uid != last_uid:
                    self._set_user_property(last_uid, uid2used_item[last_uid], positive_item)
                    last_uid = uid
                    self.uid_list.append(uid)
                    positive_item = set()
                positive_item.add(iid)
            self._set_user_property(last_uid, uid2used_item[last_uid], positive_item)
            self.uid_list = torch.tensor(self.uid_list, dtype=torch.int64)
            self.user_df = dataset.join(Interaction({self.uid_field: self.uid_list}))

        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _set_user_property(self, uid, used_item, positive_item):
        if uid is None:
            return
        history_item = used_item - positive_item
        self.uid2positive_item[uid] = torch.tensor(list(positive_item), dtype=torch.int64)
        self.uid2items_num[uid] = len(positive_item)
        self.uid2history_item[uid] = torch.tensor(list(history_item), dtype=torch.int64)

    def _init_batch_size_and_step(self):
        batch_size = self.config['eval_batch_size']
        if not self.is_sequential:
            batch_num = max(batch_size // self.dataset.item_num, 1)
            new_batch_size = batch_num * self.dataset.item_num
            self.step = batch_num
            self.set_batch_size(new_batch_size)
        else:
            self.step = batch_size
            self.set_batch_size(batch_size)

    @property
    def pr_end(self):
        if not self.is_sequential:
            return len(self.uid_list)
        else:
            return len(self.dataset)

    def _shuffle(self):
        self.logger.warnning('FullSortEvalDataLoader can\'t shuffle')

    def _next_batch_data(self):
        if not self.is_sequential:
            user_df = self.user_df[self.pr:self.pr + self.step]
            uid_list = list(user_df[self.uid_field])

            history_item = self.uid2history_item[uid_list]
            positive_item = self.uid2positive_item[uid_list]

            history_u = torch.cat([torch.full_like(hist_iid, i) for i, hist_iid in enumerate(history_item)])
            history_i = torch.cat(list(history_item))

            positive_u = torch.cat([torch.full_like(pos_iid, i) for i, pos_iid in enumerate(positive_item)])
            positive_i = torch.cat(list(positive_item))

            self.pr += self.step
            return user_df, (history_u, history_i), positive_u, positive_i
        else:
            interaction = self.dataset[self.pr:self.pr + self.step]
            inter_num = len(interaction)
            positive_u = torch.arange(inter_num)
            positive_i = interaction[self.iid_field]

            self.pr += self.step
            return interaction, None, positive_u, positive_i
