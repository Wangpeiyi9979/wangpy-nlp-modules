# encoding:utf-8
"""
@Time: 2020/3/6 10:03
@Author: Wang Peiyi
@Site :
@File : lstm.py
"""
import torch
import torch.nn as nn
class My_Lstm(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=True, num_layers=1):
        super(My_Lstm, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=bidirectional,
                            num_layers=num_layers)

    def _get_final_state(self, final_state):
        """
        由于一些seq2seq模型, decoder是单向的lstm，而encoder是双向的lstm,
        原始pytorch lstm的最终隐藏状态为(num_layers*number_dir, B, hidden_size)的Tensor,
        但是decoder要求输入为(num_layers, B, hidden_size*num_dir),
        因此这里修改pytorch lstm的最终隐藏状态，用于decoder的输入

        Hint： 原始lstm最终隐藏状态的组织形式是[第一层正向, 第一层反向, 第二层正向, 第二层方向]/[第一层,第二层,第三层]
               因此我们需要进行
        Parameters
        ----------
        final_state
            原始lstm输出的最终状态h, 为(num_layers*number_dir, B, hidden_size)的Tensor
        Returns
        -------
            经转换后的最终状态，为(num_layers, B, hidden_size*num_dir)
        """
        if self.num_direction == 1:
            return final_state
        split_state = final_state.split(self.num_direction, dim=0)
        x = list(map(lambda x: torch.cat([x[0], x[1]], 1).unsqueeze(0), split_state))
        return torch.cat(x, 0)

    def forward(self, inputs, lengths=None):

        if not lengths is None:
            sorted_lengths, sorted_indexs = torch.sort(lengths, descending=True)
            tmp1, desorted_indexs = torch.sort(sorted_indexs, descending=False)
            x_rnn = inputs[sorted_indexs]
            packed_x_rnn = nn.utils.rnn.pack_padded_sequence(x_rnn, sorted_lengths.cpu().numpy(), batch_first=True)
            packed_rnn_output, sorted_h_and_c = self.lstm(packed_x_rnn)
            sort_rnn_output, tmp3 = nn.utils.rnn.pad_packed_sequence(packed_rnn_output, batch_first=True)
            rnn_output = sort_rnn_output[desorted_indexs]  # (B, N, 2或1*hidden_size)
            finnal_h = sorted_h_and_c[0][:, desorted_indexs, :]  # (num_layers*number_dir, B, hidden_size)
            finnal_c = sorted_h_and_c[1][:, desorted_indexs, :]
        else:
            rnn_output, (finnal_h, finnal_c) = self.lstm(inputs)

        finnal_h = self._get_final_state(finnal_h)
        finnal_c = self._get_final_state(finnal_c)
        return rnn_output, (finnal_h, finnal_c)
