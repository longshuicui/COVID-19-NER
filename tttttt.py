# -*- coding: utf-8 -*-
"""
@author: longshuicui
@date  :   2020/9/12
@function:
"""
import json
import logging
from collections import defaultdict, Counter
import tokenization

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


def is_whitespace(c):
    if c==" " or c=="\r" or c=="\n" or c=="\t" or ord(c)==0x202F:
        return True
    return False


def _improve_answer_span(doc_tokens, input_start,input_end,tokenizer,origin_text):
    """实体是通过字符位置标注，bpe分词之后有比空格分词更好的匹配结果"""
    token_text=" ".join(tokenizer.tokenize(origin_text))
    for new_start in range(input_start,input_end+1):
        for new_end in range(input_end,new_start-1,-1):
            text_span=" ".join(doc_tokens[new_start:new_end+1])
            if text_span == token_text:
                return new_start,new_end


def read_task_examples(input_file,is_training,tokenizer):
    """read data"""
    def is_whitespace(c):
        if c == " " or c == "\r" or c == "\n" or c == "\t" or ord(c) == 0x202F:
            return True
        return False

    def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, origin_text):
        """实体是通过字符位置标注，bpe分词之后有比空格分词更好的匹配结果"""
        token_text = " ".join(tokenizer.tokenize(origin_text))
        for new_start in range(input_start, input_end + 1):
            for new_end in range(input_end, new_start - 1, -1):
                text_span = " ".join(doc_tokens[new_start:new_end + 1])
                if text_span == token_text:
                    return new_start, new_end

    with open(input_file,encoding="utf8") as reader:
        input_data=[json.loads(line.strip()) for line in reader]
    logging.info("*Number*:%d"%len(input_data))

    examples=[]
    for eid,entry in enumerate(input_data):
        raw_text=entry["text"]
        doc_tokens=[] # doc 按照空格分词
        char_to_word_offset=[] # 该字符对应的是那个单词，
        prev_is_whitespace=True # 前一个字符是否为空格，默认为True，起始第一个字符的前一个一定是空
        for c in raw_text:
            if is_whitespace(c):
                prev_is_whitespace=True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1]+=c
                prev_is_whitespace=False
            char_to_word_offset.append(len(doc_tokens)-1)

        # 上面是按照空格分词，下面是wordpiece分词
        token_to_origin_index=[]
        origin_to_token_index=[]
        all_doc_tokens=[]
        for i, token in enumerate(doc_tokens):
            origin_to_token_index.append(len(all_doc_tokens))
            sub_tokens=tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                token_to_origin_index.append(i)
                all_doc_tokens.append(sub_token)

        tag_labels=None
        if is_training:
            tag_labels=["O"]*len(all_doc_tokens)
            entities=entry["entities"]
            for entity in entities:
                ent_tokens=entity["entity"]
                ent_type=entity["type"]
                ent_start,ent_end=entity["start"],entity["end"]
                # 下面将字符起止位置用空格分词的token索引代替
                start_position=char_to_word_offset[ent_start]
                end_position=char_to_word_offset[ent_end]
                if raw_text.find(ent_tokens)==-1:
                    logging.info("Could not find entity: %s"%ent_tokens)
                    continue
                # 下面将起止位置用wordpiece分词之后的索引表述
                token_start_position=origin_to_token_index[start_position]
                if end_position<len(doc_tokens)-1:
                    token_end_position=origin_to_token_index[end_position+1]-1
                else:
                    token_end_position=len(all_doc_tokens)-1
                # 获取更加匹配的索引
                token_start_position,token_end_position=_improve_answer_span(all_doc_tokens,token_start_position,token_end_position,tokenizer,ent_tokens)
                tag_labels[token_start_position]="B-"+ent_type
                for index in range(token_start_position+1,token_end_position+1):
                    tag_labels[index]="I-"+ent_type

        example=InputExample(guid=eid,text_a=all_doc_tokens,label=tag_labels)

        print(example)

        exit()














if __name__ == '__main__':
    tokenizer=tokenization.FullTokenizer(vocab_file="./vocab.txt",do_lower_case=True)
    # s="cells"
    # t=tokenizer.tokenize(s)
    # print(t)
    read_task_examples("../task1_public/new_train.json",is_training=True,tokenizer=tokenizer)

