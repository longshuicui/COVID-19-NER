# -*- coding: utf-8 -*-
"""
@author: longshuicui
@date  :   2020/9/12
@function:
"""
import json
import logging
import re
from collections import defaultdict, Counter
import tokenization

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

    # with open(input_file,encoding="utf8") as reader:
    #     input_data=[json.loads(line.strip()) for line in reader]
    s={"text": "Safety and efficacy of cognitive training plus epigallocatechin-3-gallate in young adults with Down's "
               "syndrome (TESDAD): a double-blind, randomised, placebo-controlled, phase 2 trial\tEGCG and cognitive "
               "training for 12 months was significantly more effective than placebo and cognitive training at "
               "improving visual recognition memory, inhibitory control, and adaptive behaviour. Phase 3 trials "
               "with a larger population of individuals with Down's syndrome will be needed to assess and confirm "
               "the long-term efficacy of EGCG and cognitive training.",
       "entities": [{"entity": "epigallocatechin-3-gallate", "type": "ChemicalCompound", "start": 47, "end": 73},
                    {"entity": "Down's syndrome", "type": "Disease", "start": 95, "end": 110},
                    {"entity": "Down's syndrome", "type": "Disease", "start": 438, "end": 453},
                    {"entity": "EGCG", "type": "ChemicalCompound", "start": 183, "end": 187},
                    {"entity": "EGCG", "type": "ChemicalCompound", "start": 517, "end": 521}]}
    input_data=[s]
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

        # example=InputExample(guid=eid,text_a=all_doc_tokens,label=tag_labels)

        print(all_doc_tokens)
        print(tag_labels)

        exit()

def token_decode(text):
    tokens=[]
    for t in text.split():
        if t[:2] == "##":
            if len(tokens)==0:
                tokens.append(t[2:])
            else:
                tokens[-1]+=t[2:]
        else:
            tokens.append(t)
    return " ".join(tokens)

def write_predictions(origin_text_file,prediction_file,output_prediction_file):
    """将预测结果写入json文件"""
    # tokenizer = tokenization.BasicTokenizer(do_lower_case=True)
    with open(origin_text_file,encoding="utf8") as file:
        origin_texts=file.readlines()
    with open(prediction_file,encoding="utf8") as file:
        predict_texts=file.readlines()
    assert len(origin_texts)==len(predict_texts)

    writer=open(output_prediction_file,"w",encoding="utf8")

    predict_result=[]
    for i, line  in enumerate(predict_texts):
        origin_text=json.loads(origin_texts[i].strip())["text"]
        # fw_text = " ".join(tokenizer.tokenize(origin_text))

        tok_text, tok_labels=line.strip().split("\t")
        # bw_text = token_decode(tok_text)
        sub_tokens = ["[CLS]"] + tok_text.split() + ["[SEP]"]  # 分词时需要加上 [CLS] 和 [SEP] 两个标识
        sub_labels = tok_labels.split()  # 获取每个token的标签

        # 获取每条样本的实体集合，
        entities = []
        entity = ""
        entity_type = None
        for k in range(len(sub_tokens)):
            if sub_labels[k] == "O" or sub_labels[k] == "[CLS]":
                if entity!="":
                    entity=token_decode(entity)
                    entities.append({"entity": entity, "type": entity_type})
                    entity = ""
            elif sub_labels[k][0] == "B":
                entity = sub_tokens[k]
                entity_type = sub_labels[k][2:]
            elif sub_labels[k][0] == "I":
                entity += " " + sub_tokens[k]
            else:  # 当前标识为[SEP]，表示句子已经结束
                break

        lower_text=origin_text.lower()
        print(lower_text)
        current=0
        new_entities=[]
        for entity in entities:
            entity["entity"]=entity["entity"].replace(" - ","-")
            entity["entity"]=entity["entity"].replace(" ' ","'")
            start=lower_text.find(entity["entity"],current)
            if start<0:continue
            end=start+len(entity["entity"])-1
            current=end
            entity["start"]=start
            entity["end"]=end
            new_entities.append(entity)

        print(new_entities)
        for entity in new_entities:
            entity["entity"]=origin_text[entity["start"]:entity["end"]+1]

        print(new_entities)
        print()

        item=defaultdict()
        item["text"]=origin_text
        item["entities"]=new_entities if len(new_entities)>0 else [{"entity":"entity","type":"type","start":1,"end":2}]
        item=json.dumps(item,ensure_ascii=False)
        writer.write(item+"\n")



        # if i>3:
        #   exit()





    pass











if __name__ == '__main__':
    # write_predictions("./task1_public/new_val.json","./test_results.txt","./submit.json")
    tokenizer=tokenization.FullTokenizer("./vocab.txt",do_lower_case=True)
    read_task_examples("",True,tokenizer)