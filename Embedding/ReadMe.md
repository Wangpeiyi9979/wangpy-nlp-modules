# Embedding���װ
## word2vec
### ����embedder
```python
from word2vec import word2vec_Embedder
word2vec_embedder = word2vec_Embedder(word_file='./data/bert_vocab.txt',
                                word2vec_file='./data/glove.840B.300d.txt',
                                static=False,
                                use_gpu=True)
```
- `word_file`: �ض������Ӧ�Ĵʿ⣬һ��Ϊһ�����ʡ��Զ����`@UNKNOW_TOKEN@`��`@PADDING_TOKEN@`��
```
word1
word2
word3
....
```
- `word2vec_file`: �������ļ�������glove, ��https://nlp.stanford.edu/projects/word2vec/
- `static`: ��ʾ�Ƿ����word2vec embedding����
- `use_gpu`: �Ƿ�ʹ��gpu

### ʹ��embedder
```python
tokens_list=[['i','hate','this'],['i','am','your','friend']]
embedding = word2vec_embedder(tokens_list)
```
- `tokens_list:` `batch_size`���з�Ϊ�����б�ľ��ӣ�����һ����
- `embedding`: һ��`(batch_size x max_length x word_dim)`��Tensor.����Ķ�Ӧ��`@PADDING_TOKEN@`��embedding.
### ����
����ṩ���ļ�Ϊ���ļ�, �����������̬�ֲ������ʼ��, ������ģ��Ҳ����������Ϊ`pos_tag`�ȵ�embedding.
# Bert
### ����embedder
```python
from bert import Bert_Embedder
bert_embedder = Bert_Embedder(vocab_dir='./data/bert_vocab.txt',  # bert�ʱ�
                              bert_model_dir='/home/wpy/data/word2vec/bert-base-cased', # bertԤѵ��ģ�ͣ�������һ��json�ļ���һ��bin�ļ�
                              output_all_encoder_layers=False,
                              split=True,
                              use_gpu=True)
```
- `vocab_dir`: bert�ʱ�
- `bert_model_dir`: bertԤѵ��ģ�Ͳ���
- `split`: �Ƿ�������ÿһ�����ʽ������з֡�������з֣��򲻴���bert�ʱ��еĵ��ʱ��滻Ϊ`[UNK]`������з�, ��ô���ʻᱻ�з�Ϊ��С��Ԫ��
��`trainyou -> train, ##you`, ���`trainyou`��embeddingΪ`train, ##you`���ߵ�ƽ��
- `use_gpu`: �Ƿ�ʹ��gpuѵ��
### ʹ��embedder
```python
tokens_list=[['i','hate','this'],['i','am','your','friend']]
embedding, pooled_out = bert_embedder(tokens_list)
```
- ����:
    - `tokens_list:`ͬword2vec����һ��, `batch_size`���з�Ϊ�����б�ľ��ӣ�����һ����.
- ���:
    - `embedding`: 
        - ��`split`Ϊ`True`: �򷵻�`(batch_size x max_length x word_dim)`��Tensor, `[PAD]`������Ϊ0
        - ��`split`Ϊ`False`: �򷵻�`(batch_size x (max_length+2) x word_dim)`��Tensor, ÿ�����Ӷ�����`[CLS]`��`[SEP]`
        ��embedding, ����ԭʼbert��ʵ��ԭ��, �����`[PAD]`��embedding����0.
    - `pooled_out`:`Tensor(n, hidden_size`): ÿ���������һ��encoder�ĵ�һ����`[CLS]`����Linear��ͼ����`Tanh()`���Tensor. ������˾�����Ϣ
    
