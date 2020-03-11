# -*- coding:utf-8 -*-

import os
from bert import modeling
import tensorflow as tf
from bert import tokenization

flags = tf.flags
FLAGS = flags.FLAGS

bert_path = r'chinese_L-12_H-768_A-12'
root_path = os.getcwd()

flags.DEFINE_string(
    "bert_config_file", os.path.join(bert_path, 'bert_config.json'),
    "The config json file corresponding to the pre-trained BERT model."
)
flags.DEFINE_string("vocab_file", os.path.join(bert_path, 'vocab.txt'),
                    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text."
)
flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization."
)

bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

def data_preprocess(sentence):
    tokens = []
    for i, word in enumerate(sentence):
        # 分词，如果是中文，就是分字
        token = tokenizer.tokenize(word)
        tokens.extend(token)
    # 序列截断
    if len(tokens) >= FLAGS.max_seq_length - 1:
        tokens = tokens[0:(FLAGS.max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
    ntokens = []
    segment_ids = []
    ntokens.append("[CLS]")  # 句子开始设置CLS 标志
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
    ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # 将序列中的字(ntokens)转化为ID形式
    # print(input_ids)
    input_mask = [1] * len(input_ids)
    # print(input_mask)
    while len(input_ids) < FLAGS.max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
    input_ids = [input_ids]
    return input_ids, input_mask

class BertEncoder(object):

    def __init__(self):
        self.bert_model = modeling.BertModel(config=bert_config, is_training=False, max_seq_length=FLAGS.max_seq_length)
        tvars = tf.trainable_variables()
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, FLAGS.init_cheeckpoint)
        tf.train.init_from_checkpoint(FLAGS.init_cheeckpoint, assignment_map)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def encode(self, sentence):
        input_ids, input_mask = data_preprocess(sentence)
        return self.sess.run(self.bert_model.embedding_output, feed_dict={self.bert_model.input_ids:input_ids})



if __name__ == "__main__":
    be = BertEncoder()
    embedding = be.encode("新年快乐，恭喜发财，万事如意！")
    print(embedding)
    print(embedding.shape)