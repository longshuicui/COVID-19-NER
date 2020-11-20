# COVID-19 知识图谱构建-NER
https://www.biendata.xyz/competition/chaindream_knowledgegraph_19_task1

## Content
run_ner_finetune.py 序列标注方式  
run_point_finetune.py 阅读理解方式，采用半指针半标注方法进行标注（reference：https://github.com/bojone/kg-2019-baseline）  
utils.py脚本中，write_predictions_ner是序列标注的解码,write_predictions_point是阅读理解的解码

## 效果
均为single model  
NER方式，f1=0.7005  
MRC方式，f1=0.7242


## 运行
```buildoutcfg
# 将bert源码克隆到本地
git clone https://github.com/google-research/bert.git
# 下载bert预训练文件，这里用的是bert-base
wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip
# 创建一个文件夹存放下载的预训练文件
mkdir uncased_L-12_H-768_A-12 & cd uncased_L-12_H-768_A-12
# 将下载的文件移动到刚创建的文件夹
mv ../uncased_L-12_H-768_A-12.zip ./
# 解压
unzip uncased_L-12_H-768_A-12.zip
# 返回上一级目录
cd ..
# 运行脚本
python run_ner_finetune.py
# 结果解码，生成submit.json, 根据训练方式不同选择不同的解码方式
python utils.py
```
