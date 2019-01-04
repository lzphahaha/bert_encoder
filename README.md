# bert_encoder
use Google Bert model to encode a sentence to vector.

**usage**

**[`BERT-Base, Chinese`](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)**:
    Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M
    parameters

Download the model above, unzip and place in current directory.

**How to encode a sentence?**

```
from bert_encoder import BertEncoder
be = BertEncoder()
embedding = be.encode("新年快乐，恭喜发财，万事如意！")
print(embedding)
print(embedding.shape)
```