# Embedding���װ
## glove
### ����embedder
```python
from glove import Glove_Embedder
glove_embedder = Glove_Embedder(word_file='./data/bert_vocab.txt',
                                glove_file='./data/glove.840B.300d.txt',
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
- `glove_file`: glove�������ļ�����https://nlp.stanford.edu/projects/glove/
- `static`: ��ʾ�Ƿ����glove embedding����
- `use_gpu`: �Ƿ�ʹ��gpu

### ʹ��embedder
```python
tokens_list=[['i','hate','this'],['i','am','your','friend']]
embedding = glove_embedder(tokens_list)
```
- `tokens_list:` `batch_size`���з�Ϊ�����б�ľ��ӣ�����һ����
- `embedding`: һ��`(batch_size x max_length x word_dim)`��Tensor.����Ķ�Ӧ��`@PADDING_TOKEN@`��embedding.
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
    - `tokens_list:`ͬglove����һ��, `batch_size`���з�Ϊ�����б�ľ��ӣ�����һ����.
- ���:
    - `embedding`: 
        - ��`split`Ϊ`True`: �򷵻�`(batch_size x max_length x word_dim)`��Tensor, `[PAD]`������Ϊ0
        - ��`split`Ϊ`False`: �򷵻�`(batch_size x (max_length+2) x word_dim)`��Tensor, ÿ�����Ӷ�����`[CLS]`��`[SEP]`
        ��embedding, ���ڶ����`[PAD]`��embedding����0.
    - `pooled_out`:`Tensor(n, hidden_size`): ÿ���������һ��encoder�ĵ�һ����`[CLS]`����Linear��ͼ����`Tanh()`���Tensor. ������˾�����Ϣ
    
