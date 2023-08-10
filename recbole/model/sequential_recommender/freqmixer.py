# @Time   : 2022/2/13
# @Author : Hui Yu
# @Email  : ishyu@outlook.com

import torch
import torch.nn as nn
from recbole.model.loss import BPRLoss
from recbole.model.abstract_recommender import SequentialRecommender

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class MlpMixer(nn.Module):
    def __init__(self, kept_dim, input_dim, hidden_size_factor):
        super(MlpMixer, self).__init__()
        # kept_dim代表有独立权重的那个维度的维数
        self.kept_dim = kept_dim
        self.input_dim = input_dim
        self.hidden_size_factor = hidden_size_factor
        self.w1 = nn.Parameter(torch.randn(2,
                                           self.kept_dim,
                                           self.input_dim,
                                           self.input_dim * self.hidden_size_factor) * 0.02)
        self.b1 = nn.Parameter(torch.randn(2, self.kept_dim, self.input_dim * self.hidden_size_factor) * 0.02)
        self.w2 = nn.Parameter(torch.randn(2,
                                           self.kept_dim,
                                           self.input_dim * self.hidden_size_factor,
                                           self.input_dim,) * 0.02)
        self.b2 = nn.Parameter(torch.randn(2, self.kept_dim, self.input_dim) * 0.02)

    def forward(self, x):
        x_real, x_image = torch.real(x), torch.imag(x)
        # 以frequency_mixing为例子解释下面的einsum
        # o1作为复数，计算公式为o1 = GeLU([x_real + i * x_image] * [w1_real + i * w1_iamge] + [b1_real + i * b1_image])
        # = GeLU(x_real * w1_real - x_image * w1_image + b1_real) + i * GeLU(x_image * w1_real + x_image * w1_real + b1_image)
        # frequency_mixing对不同的hidden_dimension共享mlp权重，但是对不同的segment设置了独立的mlp权重，所以
        # w1.shape = [2, segment_num, frequency_num, frequency_num * 2]，其中2分别为复数的实部和虚部，segment_num为序列片段的数目，
        # frequency_num为不同频率的数目，frequency_num * 2为隐藏层维度
        # 在计算o1_real时'...ijk,jkl->...ijl'表示'...(hidden_dimension)(segment)(frequency_num),(segment)(frequency_num)(frequency_num * 2)->...(hidden_dimension)(segment)(frequency_num * 2)'
        # 也就是输入序列的实部的shape=[batch_size, hidden_size, segment_num, frequency_num], w1_real.shape=[segment_num, frequency_num, frequency_num * 2]
        # einsum对输入序列x的不同的segment用来自w1_real的不同的segment来做mlp，但不同的hidden_size维度共享权重
        o1_real = F.gelu(
            torch.einsum('...ijk,jkl->...ijl', x_real, self.w1[0]) -
            torch.einsum('...ijk,jkl->...ijl', x_image, self.w1[1]) +
            self.b1[0]
        )
        o1_image = F.gelu(
            torch.einsum('...ijk,jkl->...ijl', x_image, self.w1[0]) +
            torch.einsum('...ijk,jkl->...ijl', x_real, self.w1[1]) +
            self.b1[1]
        )
        o2_real = (
            torch.einsum('...ijk,jkl->...ijl', o1_real, self.w2[0]) -
            torch.einsum('...ijk,jkl->...ijl', o1_image, self.w2[1]) +
            self.b2[0]
        )
        o2_image = (
            torch.einsum('...ijk,jkl->...ijl', o1_image, self.w2[0]) +
            torch.einsum('...ijk,jkl->...ijl', o1_image, self.w2[1]) +
            self.b2[1]
        )
        x = torch.stack([o2_real, o2_image], dim=-1)
        x = torch.view_as_complex(x)
        return x


class FrequencyDomainMLP(nn.Module):
    def __init__(self, config):
        super(FrequencyDomainMLP, self).__init__()
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']
        self.hidden_size = config['hidden_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.hidden_size_factor = config['hidden_size_factor']

        self.segment_len = config['segment_len']
        self.hop_len = config['hop_len']
        self.frequency_num = self.segment_len // 2 + 1
        self.segment_num = self.max_seq_length // self.hop_len + 1
        self.pad_seq_len = self.segment_num * self.hop_len if self.max_seq_length % self.hop_len != 0 else self.max_seq_length
        self.segment_num = self.segment_num + 1 if self.pad_seq_len > self.max_seq_length else self.segment_num
        self.pad_len = self.pad_seq_len - self.max_seq_length
        self.zero_pad = nn.ZeroPad2d((0, self.pad_len, 0, 0))

        self.frequency_mixing = MlpMixer(self.segment_num, self.frequency_num, self.hidden_size_factor)
        self.channel_mixing = MlpMixer(self.segment_num, self.hidden_size, self.hidden_size_factor)
        self.segment_mixing = MlpMixer(self.frequency_num, self.segment_num, self.hidden_size_factor)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.LayerNorm = LayerNorm(self.hidden_size)
        self.feedforward = FeedForward(self.hidden_size, self.hidden_dropout_prob)
        self.conv2d = nn.Conv2d(self.segment_num,
                                self.segment_num,
                                (self.hidden_size, self.frequency_num),
                                groups=self.segment_num)
        self.exp_decay_bias = torch.exp(torch.tensor(range(1-self.segment_num, 1), dtype=torch.float)).cuda()

    def forward(self, input_tensor):
        batch, seq_len, hidden = input_tensor.shape
        # x.shape = [batch_size, hidden_size, seq_len]
        x = input_tensor.transpose(-1, -2)
        # x.shape = [batch_size, hidden_size, pad_seq_len]
        x = self.zero_pad(x)
        # x.shape = [batch_size * hidden_size, pad_seq_len]
        x = x.contiguous().view(batch * hidden, self.pad_seq_len)
        # x.shape = [batch_size * hidden_size, frequency_num, segment_num, 2], frequency_num = segment_len // 2 + 1, 最后一个维度的2代表复数的实部和虚部
        x = torch.stft(x, n_fft=self.segment_len, hop_length=self.hop_len, center=True, onesided=True, normalized=True, pad_mode='constant')
        x = x.reshape(batch, hidden, self.frequency_num, self.segment_num, 2)
        xnorm = x.norm(2, -1).reshape(batch, hidden, self.frequency_num, self.segment_num)
        weight = self.conv2d(xnorm.permute(0, 3, 1, 2).contiguous())
        weight = weight.permute(0, 2, 3, 1) + self.exp_decay_bias
        # x.shape = [batch_size, hidden_size, frequency_num, segment_num], x.dtype = complex
        x = torch.view_as_complex(x) * weight
        # x.shape = [batch_size, hidden_size, segment_num, frequency_num]
        x = x.transpose(-1, -2)
        # residual.shape = [batch_size, hidden_size, segment_num, frequency_num]
        residual = self.frequency_mixing(x) #在frequency mixing中，不同的hidden dimension共享权重，不同的segment有独立的权重
        x = residual
        # x.shape = [batch_size, frequency_num, segment_num, hidden_size]
        x = x.transpose(1, 3)
        # residual.shape = [batch_size, frequency_num, segment_num, hidden_size]
        residual = self.channel_mixing(x) #在channel mixing中，不同的hidden dimension共享权重，不同的segment有独立的参数
        x = residual
        # x.shape = [batch_size, hidden_size, frequency_num, segment_num]
        x = x.permute(0, 3, 1, 2)
        # residual.shape = [batch_size, hidden_size, frequency_num, segment_num]
        residual = self.segment_mixing(x) #在segment mixing中，不同的hidden dimension共享权重，不同的frequency有独立权重
        x = residual
        # x.shape = [batch_size * hidden_size, frequency_num, segment_num]
        x = x.view(batch * hidden, self.frequency_num, self.segment_num)
        # x.shape = [batch_size * hidden_size, frequency_num, segment_num, 2]
        x = torch.view_as_real(x)
        # x.shape = [batch_size * hidden_size, pad_seq_len]
        x = torch.istft(x,
                        n_fft=self.segment_len,
                        hop_length=self.hop_len,
                        center=True,
                        onesided=True,
                        normalized=True)
        # x.shape = [batch_size * hidden_size, max_seq_len]
        x = x[:, :seq_len]
        # x.shape = [batch_size, hidden_size, max_seq_len]
        x = x.view(batch, hidden, seq_len)
        # x.shape = [batch_size, max_seq_len, hidden_size]
        x = x.transpose(-1, -2)
        hidden_states = self.dropout(x)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        hidden_states = self.feedforward(hidden_states)
        return hidden_states


class ItemPredictor(nn.Module):
    def __init__(self, max_seq_len, hidden_size, hidden_size_fractor):
        super(ItemPredictor, self).__init__()
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.hidden_size_factor = hidden_size_fractor
        self.conv1 = nn.Linear(self.max_seq_len, 1)
        self.b = nn.Parameter(torch.randn(1) * 0.02)
        self.mlp = nn.Sequential(nn.Linear(self.max_seq_len, self.max_seq_len * self.hidden_size_factor),
                                 nn.GELU(),
                                 nn.Linear(self.max_seq_len * self.hidden_size_factor, self.max_seq_len))

    def forward(self, inputs):
        x = inputs.transpose(-1, -2)
        x = self.conv1(x)
        x = x.transpose(-1, -2).squeeze(1)
        x = x.softmax(dim=-1)
        alpha = torch.bmm(inputs, x.unsqueeze(-1)).squeeze(-1) + self.b
        alpha = alpha.squeeze(-1)
        alpha = self.mlp(alpha)
        alpha = alpha.unsqueeze(-1).to(torch.double)
        return alpha


class FeedForward(nn.Module):
    def __init__(self, hidden_size, dropout_prob):
        super(FeedForward, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_dropout_prob = dropout_prob
        self.dense_1 = nn.Linear(self.hidden_size, self.hidden_size * 4)
        self.intermediate_act_fn = nn.GELU()
        self.dense_2 = nn.Linear(4 * self.hidden_size, self.hidden_size)
        self.LayerNorm = LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class FreqMixer(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(FreqMixer, self).__init__(config, dataset)

        # load parameters info
        self.n_mixers = config['n_mixers']
        self.config = config
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.hidden_size_factor = config['hidden_size_factor']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.item_predictor = ItemPredictor(self.max_seq_length, self.hidden_size, self.hidden_size_factor)
        self.seq_dropout = nn.Dropout(0.2)
        self.item_dropout = nn.Dropout(0.2)
        self.temperature = config['temperature']
        self.fn = nn.Linear(self.hidden_size, self.hidden_size)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.batch_size = config['train_batch_size']
        self.mixers = nn.Sequential(*[FrequencyDomainMLP(config) for _ in range(self.n_mixers)])

        # parameters initialization
        self.apply(self._init_weights)

    def device_as(self, t1, t2):
        """
        Moves t1 to the device of t2
        """
        return t1.to(t2.device)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, return_embedding=False, return_alpha=False, return_filter=False):
        filter_weight = []
        item_emb = self.item_embedding(item_seq)
        item_emb = self.seq_dropout(item_emb)
        if return_embedding:
            item_emb.retain_grad()
        output = item_emb
        for i in range(self.n_mixers):
            output = self.mixers[i](output)
        if return_filter:
            return filter_weight
        # return output
        alpha = self.item_predictor(output)
        mask = item_seq.gt(0)
        alpha = torch.where(mask.unsqueeze(-1), alpha, -9e15)
        alpha = torch.softmax(alpha, dim=1, dtype=torch.float)
        seq_output = torch.sum(alpha * output, dim=1)
        seq_output = self.fn(seq_output)
        seq_output = F.normalize(seq_output, dim=-1)

        if return_embedding:
            return seq_output, item_emb
        if return_alpha:
            return seq_output, alpha
        else:
            return seq_output, None  # [B H]

    def calculate_loss(self, interaction, knn_negative_samples=None, knn_items_sample=None, optimizer=None):
        item_seq = interaction[self.ITEM_SEQ]  # N * L
        seq_output, alpha = self.forward(item_seq)  # N * D
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
        elif self.loss_type == 'CE':  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            test_item_emb = self.item_dropout(test_item_emb)
            test_item_emb = F.normalize(test_item_emb, dim=-1)
            logit = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) / self.temperature
            loss = self.loss_fct(logit, pos_items)
        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output, _ = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        seq_output, _ = self.forward(item_seq)
        test_items_emb = self.item_embedding.weight
        test_items_emb = F.normalize(test_items_emb, dim=-1)
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1)) / self.temperature  # [B n_items]
        return scores

    def get_seq_representation(self, item_seq):
        item_emb = self.item_embedding(item_seq)
        item_emb = self.seq_dropout(item_emb)
        output = self.forward(item_seq)
        return output

    def get_filters(self, interaction):
        item_seq = interaction[self.ITEM_SEQ].cuda()
        item_seq_len = interaction[self.ITEM_SEQ_LEN].cuda()
        user_id_seq = interaction['user_id'].cuda()
        rating_seq = interaction['rating_list'].cuda()
        filter_weight = self.forward(item_seq)
        filters = []
        for i in range(len(filter_weight)):
            f = torch.stack([torch.real(filter_weight[i]), torch.imag(filter_weight[i])], dim=-1)
            f /= f.min()
            f = f.norm(dim=-1).cpu().detach().clone().numpy()
            filters.append(f)
        return filters



