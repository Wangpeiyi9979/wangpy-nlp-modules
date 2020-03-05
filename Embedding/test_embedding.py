#encoding:utf-8
"""
@Time: 2020/3/4 12:57
@Author: Wang Peiyi
@Site : 
@File : test_embedding.py
"""
tokens_list = """i hate this
   i am your friend""".lower().split(
    '\n')
tokens_list = list(map(lambda x: x.split(), tokens_list))

# 使用glove embedding
def test_glove():
    from glove import Glove_Embedder
    glove_embedder = Glove_Embedder(word_file='./data/bert_vocab.txt',
                                    glove_file='./data/glove.840B.300d.txt',  # glove词表
                                    static=False,
                                    use_gpu=True)
    glove_embedder = glove_embedder.cuda()
    embedding = glove_embedder(tokens_list)
    print(embedding.size())

def test_bert():
    from bert import Bert_Embedder
    bert_embedder = Bert_Embedder(vocab_dir='./data/bert_vocab.txt',  # bert词表
                                  bert_model_dir='/home/wpy/data/word2vec/bert-base-cased', # bert预训练模型，里面有一个json文件和一个bin文件
                                  output_all_encoder_layers=False,
                                  split=True,
                                  use_gpu=True)
    bert_embedder = bert_embedder.cuda()
    embedding, pooled_out = bert_embedder(tokens_list)
    print(embedding.size())
test_glove()
test_bert()
