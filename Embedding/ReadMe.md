# Embedding层封装
## glove
### 创建embedder
```python
from glove import Glove_Embedder
glove_embedder = Glove_Embedder(word_file='./data/bert_vocab.txt',
                                glove_file='./data/glove.840B.300d.txt',
                                static=False,
                                use_gpu=True)
```
- `word_file`: 特定任务对应的词库，一行为一个单词。自动添加`@UNKNOW_TOKEN@`和`@PADDING_TOKEN@`。
    ```
    word1
    word2
    word3
    ....
    ```
- `glove_file`: glove词向量文件。见https://nlp.stanford.edu/projects/glove/
- `static`: 表示是否更新glove embedding参数
- `use_gpu`: 是否使用gpu

### 使用embedder
```python
tokens_list=[['i','hate','this'],['i','am','your','friend']]
embedding = glove_embedder(tokens_list)
```
- `tokens_list:` `batch_size`个切分为单词列表的句子，不用一样长
- `embedding`: 一个`(batch_size x max_length x word_dim)`的Tensor.多余的对应着`@PADDING_TOKEN@`的embedding.
# Bert
### 创建embedder
```python
from bert import Bert_Embedder
bert_embedder = Bert_Embedder(vocab_dir='./data/bert_vocab.txt',  # bert词表
                              bert_model_dir='/home/wpy/data/word2vec/bert-base-cased', # bert预训练模型，里面有一个json文件和一个bin文件
                              output_all_encoder_layers=False,
                              split=True,
                              use_gpu=True)

```
- `vocab_dir`: bert词表
- `bert_model_dir`: bert预训练模型参数
- `split`: 是否对输入的每一个单词进行再切分。如果不切分，则不存在bert词表中的单词被替换为`[UNK]`。如果切分, 那么单词会被切分为更小单元，
如`trainyou -> train, ##you`, 最后`trainyou`的embedding为`train, ##you`两者的平均
- `use_gpu`: 是否使用gpu训练
### 使用embedder
```python
tokens_list=[['i','hate','this'],['i','am','your','friend']]
embedding, pooled_out = bert_embedder(tokens_list)
```
- 输入:
    - `tokens_list:`同glove输入一样, `batch_size`个切分为单词列表的句子，不用一样长.
- 输出:
    - `embedding`: 
        - 若`split`为`True`: 则返回`(batch_size x max_length x word_dim)`的Tensor, `[PAD]`的向量为0
        - 若`split`为`False`: 则返回`(batch_size x (max_length+2) x word_dim)`的Tensor, 每个句子都多了`[CLS]`和`[SEP]`
        的embedding, 由于多余的`[PAD]`的embedding不是0.
    - `pooled_out`:`Tensor(n, hidden_size`): 每个句子最后一层encoder的第一个词`[CLS]`经过Linear层和激活函数`Tanh()`后的Tensor. 其代表了句子信息
    
