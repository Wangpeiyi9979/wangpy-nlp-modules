# encoding:utf-8
"""
@Time: 2020/3/4 11:09
@Author: Wang Peiyi
@Site : 
@File : glove.py
"""
import math
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import multiprocessing


class Glove_Embedder(nn.Module):

    def __init__(self, word_file: str, glove_file: str, static=False, use_gpu=True, UNKNOW_TOKEN='@UNKNOW@',
                 PADDING_TOKEN='@PADDING@'):
        """
        @param word_file: 存储具体任务单词的txt文件，每一行是一个单词
        @param glove_file: 原始的glove文件，如"glove.840B.300d.txt"
        @param static: 表示是否更新glove embedding的参数，默认更新
        @param use_gpu: 是否使用gpu
        @param UNKNOW_TOKEN: 代表UNKONW单词
        @param PADDING_TOKEN: 代表PADDING单词
        """
        super(Glove_Embedder, self).__init__()
        self.left_word_num = 0  # 用来glove中丢失的原始vocab单词数
        self.UNKNOW_TOKEN = UNKNOW_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN
        self.static = static
        self.use_gpu = use_gpu
        self.vocab_size, self.word2id, self.id2word = self._get_vocab_size_and_word2id(word_file)
        self.word_dim, self.embedder = self._get_word_dim_and_embedder(glove_file)

        self.report_info()

    def report_info(self):
        print(
            "glove embedder构建完成, glove丢失单词数:{}/{}, 是否更新glove embedding: {}\n传入token列表得到词向量, 如[['i','hate','this'],['i','am','your','friend']]".format(
                self.left_word_num, self.vocab_size, not self.static))

    def _get_vocab_size_and_word2id(self, word_file: str):
        """
        @param word_file: 见__init__参数word_file
        @return:
            vocab_size: int, 词典大小
            word2id: dict[str: int], 映射单词到id的字典
            id2word: dict[int: str]
        """
        word2id = {self.PADDING_TOKEN: 0,
                   self.UNKNOW_TOKEN: 1}
        with open(word_file, 'r') as f:
            words = f.readlines()
            words = set(words)
            for idx, word in enumerate(words):
                word2id[word.strip()] = idx + 2
        id2word = {k: v for v, k in word2id.items()}
        return len(word2id), word2id, id2word

    def _parse_glove_lines(self, lines):
        """
        提取出glove的词向量
        @param lines:
        @return: word2vec
        """
        word2vec = {}
        for line in lines:
            word, vec = line.split(' ', 1)
            vec = vec.strip().split(' ')
            vec = np.array(list(map(lambda x: float(x), vec)))
            word2vec[word] = vec
        return word2vec

    def _get_word_dim_and_embedder(self, glove_file: str):
        """
        @param glove_file: 见__init__参数glove_file
        @return:
            word_dim(int): 单词的embedding维数
            embedder:(nn.Embedding), 这里就是用glove_file去初始化nn.Embedding的look up table
        """

        with open(glove_file, 'r') as f:
            lines = f.readlines()

        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        p = multiprocessing.Pool(cpu_count)
        process_reses = []
        print("多进程读取glove文件(cpu核数:{})..".format(cpu_count))

        size = math.ceil(len(lines) / cpu_count)
        for index in range(cpu_count):
            start = index * size
            end = min((index + 1) * size, len(lines))
            sub_data = lines[start:end]
            process_reses.append(p.apply_async(self._parse_glove_lines, args=(sub_data,)))
        p.close()
        p.join()
        print("读取完成..")
        word2vec = {}
        for res in process_reses:
            word2vec.update(res.get())

        word_dim = len(list(word2vec.values())[0])
        look_up_table = []
        print("构造vocab中单词的look up table..")
        for id in tqdm(range(self.vocab_size)):
            word = self.id2word[id]
            vec = word2vec.get(word, None)  # 单词如果不在glove中，随机初始化一个向量
            if vec is None:
                vec = np.random.randn(word_dim)
                self.left_word_num += 1
            look_up_table.append(vec)

        look_up_table = torch.from_numpy(np.array(look_up_table))
        if self.use_gpu:
            look_up_table.cuda()
        embedder = nn.Embedding(self.vocab_size, word_dim)
        embedder.weight.data.copy_(look_up_table)

        if self.static is True:
            embedder.weight.requires_grad = False

        return word_dim, embedder

    def forward(self, tokens_lists):
        """
        @param tokenss(n*list:str): n个句子, 输入的相当于一个batch
        @return:
            embeddings:Tensor(n, max_length, word_dim), 输出n个句子中token的embedding
        """
        max_len = max(map(lambda x: len(x), tokens_lists))
        # 将句子单词装换为单词id表示
        tokens_id_lists = list(
            map(lambda x: list(map(lambda w: self.word2id.get(w, self.word2id[self.UNKNOW_TOKEN]), x)),
                tokens_lists))
        # 按最长句子进行padding
        tokens_padding_id_lists = list(
            map(lambda x: x + [self.word2id[self.PADDING_TOKEN]] * (max_len - len(x)), tokens_id_lists))
        tokens_padding_id_lists = torch.LongTensor(tokens_padding_id_lists)
        if self.use_gpu is True:
            tokens_padding_id_lists = tokens_padding_id_lists.cuda()
        embeddings = self.embedder(tokens_padding_id_lists)
        return embeddings
