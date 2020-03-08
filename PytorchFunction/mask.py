#encoding:utf-8
"""
@Time: 2020/3/8 13:27
@Author: Wang Peiyi
@Site : 
@File : mask.py
"""
import torch
def sequence_mask(sequence_length, max_len=None):
    """
    @param sequence_length: LongTensor(L):
    @param max_len: 返回mask矩阵的长度，如果不指定，则为给定所有长度的最大长度
    @return: mask: (L, max_len)
    """
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.LongTensor(range(0, max_len))
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand

if __name__ == '__main__':
    sample = torch.LongTensor([4,5,6])
    print(sample)
    print(sequence_mask(sample).float())