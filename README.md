# COVID-19 知识图谱构建-NER
https://www.biendata.xyz/competition/chaindream_knowledgegraph_19_task1

## Content
run_ner_finetune.py 序列标注方式  
run_point_finetune.py 阅读理解方式，采用半指针半标注方法进行标注（refrence：https://github.com/bojone/kg-2019-baseline）  
utils.py脚本中，write_predictions_ner是序列标注的解码,write_predictions_point是阅读理解的解码

## 效果
均为single model  
序列标注方式，f1=0.7005  
阅读理解方式，f1=0.7108

## 存在问题
阅读理解方式mask部分都会被预测成实体，虽然在输出的时候可以mask掉，暂时不知道为什么会出现这种情况