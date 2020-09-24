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

type2idx={"O":0,"Drug": 1,"Disease": 2,"Gene": 3,"ChemicalCompound": 4,"Virus": 5,"Chemical": 6,"Phenotype": 7,"Organization": 8}

def read_task_examples(s,is_training,tokenizer):
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

        tag_labels=[type2idx["O"]]*len(all_doc_tokens)
        if is_training:
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
                # tag_labels[token_start_position]="B-"+ent_type
                # for index in range(token_start_position+1,token_end_position+1):
                #     tag_labels[index]="I-"+ent_type
                tag_labels[token_start_position]=type2idx[ent_type]
                tag_labels[token_end_position]=type2idx[ent_type]

        # example=InputExample(guid=eid,text_a=all_doc_tokens,label=tag_labels)

        print(all_doc_tokens)
        print(entities)
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

def write_predictions_ner(origin_text_file,prediction_file,output_prediction_file):
    """将预测结果写入json文件"""
    # tokenizer = tokenization.BasicTokenizer(do_lower_case=True)
    with open(origin_text_file,encoding="utf8") as file:
        origin_texts=file.readlines()
    with open(prediction_file,encoding="utf8") as file:
        predict_texts=file.readlines()
    assert len(origin_texts)==len(predict_texts)

    writer=open(output_prediction_file,"w",encoding="utf8")

    for i, line  in enumerate(predict_texts):
        origin_text=json.loads(origin_texts[i].strip())["text"]
        tok_text, tok_labels=line.strip().split("\t")
        sub_tokens = ["[CLS]"] + tok_text.split() + ["[SEP]"]  # 分词时需要加上 [CLS] 和 [SEP] 两个标识
        sub_labels = tok_labels.split()  # 获取每个token的标签

        # 获取每条样本的实体集合，
        entities = []
        entity_ = ""
        entity_type = None
        for k in range(len(sub_tokens[:384])):
            if sub_labels[k] == "O" or sub_labels[k] == "[CLS]":
                if entity_!="":
                    entity_=token_decode(entity_)
                    entities.append({"entity": entity_, "type": entity_type})
                    entity_ = ""
            elif sub_labels[k][0] == "B":
                entity_ = sub_tokens[k]
                entity_type = sub_labels[k][2:]
            elif sub_labels[k][0] == "I":
                entity_ += " " + sub_tokens[k]
            else:  # 当前标识为[SEP]，表示句子已经结束
                break

        # 实体去重
        unique_entities=[]
        for entity in entities:
            if entity not in unique_entities:
                unique_entities.append(entity)

        del entities

        lower_text=origin_text.lower()
        new_entities=[]
        for entity in unique_entities:
            ent_name = entity["entity"].replace(" - ","-").replace(" ' ","'")
            current=0
            while True:
                if len(ent_name)<2:break
                start=lower_text.find(ent_name,current)
                if start<0:break
                end=start+len(ent_name)
                current=end
                new_entities.append({"entity":ent_name,"type":entity["type"] if entity["type"] else "type","start":start,"end":end})

        for entity in new_entities:
            entity["entity"]=origin_text[entity["start"]:entity["end"]]

        item=defaultdict()
        item["text"]=origin_text
        item["entities"]=new_entities if len(new_entities)>0 else [{"entity":"entity","type":"type","start":1,"end":2}]
        item=json.dumps(item,ensure_ascii=False)
        writer.write(item+"\n")

def merge_result(res1,res2,output_file):
    with open(res1,encoding="utf8") as file:
        data1=file.readlines()
    with open(res2,encoding="utf8") as file:
        data2=file.readlines()

    assert len(data1)==len(data2)

    pred_file=open(output_file,"w",encoding="utf8")

    for i in range(len(data1)):
        example_a=json.loads(data1[i].strip())
        example_b=json.loads(data2[i].strip())

        text=example_a["text"]
        entities_a=example_a["entities"]
        entities_b=example_b["entities"]

        ent_set_a=set([entity["entity"] for entity in entities_a])
        ent_set_b=set([entity["entity"] for entity in entities_b])
        for ent in ent_set_a:
            if ent not in ent_set_b:
                for entity in entities_a:
                    if entity["entity"]==ent:
                        entities_b.append(entity)

        item={"text":text,"entities":entities_b}
        pred_file.write(json.dumps(item,ensure_ascii=False)+"\n")

    pred_file.close()

def write_predictions_point(origin_text_file,prediction_file,output_prediction_file):
    with open(origin_text_file,encoding="utf8") as file:
        origin_texts=file.readlines()
    with open(prediction_file,encoding="utf8") as file:
        predict_texts=file.readlines()
    assert len(origin_texts)==len(predict_texts)

    writer=open(output_prediction_file,"w",encoding="utf8")

    for i, line in enumerate(predict_texts):
        origin_text=json.loads(origin_texts[i].strip())["text"]
        tok_text, start_labels, end_labels =line.strip().split("\t")
        sub_tokens = ["[CLS]"] + tok_text.split() + ["[SEP]"]
        sub_start_labels=start_labels.split()
        sub_end_labels=end_labels.split()
        if i==21:
            print(len(sub_tokens),sub_tokens)
            print(len(sub_start_labels),sub_start_labels)
            print(len(sub_end_labels),sub_end_labels)
        # assert len(sub_tokens)==len(sub_start_labels)==len(sub_end_labels)

        entities=[]
        entity=[]
        for index in range(min(len(sub_tokens),len(sub_start_labels),len(sub_end_labels))):
            start_label=sub_start_labels[index]
            end_label=sub_end_labels[index]
            if start_label=="O" and end_label=="O" and len(entity)==0: continue

            if start_label==end_label and start_label!="O":
                entities.append((sub_tokens[index],start_label))
            else:
                if len(entity)==0 and start_label!="O" and end_label=="O":
                    entity.append(sub_tokens[index])
                elif len(entity)!=0 and start_label=="O" and end_label=="O":
                    entity.append(sub_tokens[index])
                elif len(entity)!=0 and start_label=="O" and end_label!="O":
                    entity.append(sub_tokens[index])
                    text = token_decode(" ".join(entity))
                    entities.append((text, end_label))
                    entity = []

        unique_entities=[]
        for entity in entities:
            if entity not in unique_entities:
                unique_entities.append(entity)
        del entities

        lower_text = origin_text.lower()
        new_entities = []
        for ent_name, ent_type in unique_entities:
            ent_name=ent_name.replace(" - ","-").replace(" ' ","'")
            current=0
            while True:
                if len(ent_name)<2:break
                start=lower_text.find(ent_name,current)
                if start<0:break
                end=start+len(ent_name)
                current=end
                new_entities.append({"entity":ent_name,"type":ent_type if ent_type else "type","start":start,"end":end})

        for entity in new_entities:
            entity["entity"]=origin_text[entity["start"]:entity["end"]]

        item = defaultdict()
        item["text"] = origin_text
        item["entities"] = new_entities if len(new_entities) > 0 else [
            {"entity": "entity", "type": "type", "start": 1, "end": 2}]
        item = json.dumps(item, ensure_ascii=False)
        writer.write(item + "\n")








if __name__ == '__main__':
    write_predictions_point("./task1_public/new_val.json","./test_results.txt","./submit.json")
