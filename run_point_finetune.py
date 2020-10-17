# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import collections
import warnings
import csv
import os
import json
import random
import logging
import tf_metrics
from bert import modeling
from bert import optimization
from bert import tokenization
from lstm_crf_layer import BLSTM_CRF
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers

warnings.filterwarnings("ignore")
flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string("train_file", "task1_public/train.json","The train file")

flags.DEFINE_string("dev_file", "task1_public/dev.json","The dev or predict file")

flags.DEFINE_string("infer_file", "task1_public/new_val.json","The dev or predict file")

flags.DEFINE_string("bert_config_file", "uncased_L-12_H-768_A-12/bert_config.json", "The config json file")

flags.DEFINE_string("vocab_file", "uncased_L-12_H-768_A-12/vocab.txt", "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string("output_dir", "./output_dir","The output directory where the model checkpoints will be written.")

flags.DEFINE_string("ensemble_dir", "./ensemble_model","The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string("init_checkpoint", "uncased_L-12_H-768_A-12/bert_model.ckpt", "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool("do_lower_case", True, "Whether to lower case the input text.")

flags.DEFINE_integer("max_seq_length", 512, "The maximum total input sequence length")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", True, "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 8, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 15.0, "Total number of training epochs to perform.")

flags.DEFINE_float("warmup_proportion", 0.1, "Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000, "How often to save the model checkpoint.")


class InputExample(object):
    def __init__(self, guid, tokens, start_label=None, end_label=None):
        self.guid = guid
        self.tokens = tokens
        self.start_label = start_label
        self.end_label = end_label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_label_id,
                 end_label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_label_id = start_label_id
        self.end_label_id = end_label_id
        self.is_real_example = is_real_example


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, origin_text):
    """实体是通过字符位置标注，bpe分词之后有比空格分词更好的匹配结果"""
    token_text = " ".join(tokenizer.tokenize(origin_text))
    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:new_end + 1])
            if text_span == token_text:
                return new_start, new_end


def read_task_examples(input_file,is_training,tokenizer,type2idx):
    """read data"""
    def is_whitespace(c):
        if c == " " or c == "\r" or c == "\n" or c == "\t" or ord(c) == 0x202F:
            return True
        return False

    with open(input_file,encoding="utf8") as reader:
        input_data=[json.loads(line.strip()) for line in reader]

    examples=[]
    error_num=0
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

        tag_start_labels=[type2idx["O"]]*len(all_doc_tokens)
        tag_end_labels=[type2idx["O"]]*len(all_doc_tokens)
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
                    continue
                # 下面将起止位置用wordpiece分词之后的索引表述
                token_start_position=origin_to_token_index[start_position]
                if end_position<len(doc_tokens)-1:
                    token_end_position=origin_to_token_index[end_position+1]-1
                else:
                    token_end_position=len(all_doc_tokens)-1
                # 获取更加匹配的索引
                try:
                    token_start_position,token_end_position=_improve_answer_span(all_doc_tokens,token_start_position,token_end_position,tokenizer,ent_tokens)
                except TypeError:
                    error_num+=1
                # 将label转换为类别id
                tag_start_labels[token_start_position] = type2idx[ent_type]
                tag_end_labels[token_end_position] = type2idx[ent_type]

        example=InputExample(guid=eid,tokens=all_doc_tokens,start_label=tag_start_labels,end_label=tag_end_labels)
        examples.append(example)

    #random.shuffle(examples)
    print("错误个数：",error_num)
    return examples


def convert_single_example(ex_index, example, type2idx, max_seq_length, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    raw_tokens=example.tokens
    start_labels=example.start_label
    end_labels=example.end_label
    if len(raw_tokens) > max_seq_length - 2:
        raw_tokens = raw_tokens[0:(max_seq_length - 2)]
        start_labels = start_labels[0:(max_seq_length - 2)]
        end_labels = end_labels[0:(max_seq_length - 2)]

    tokens = []
    start_label_ids=[]
    end_label_ids=[]
    segment_ids = []
    tokens.append("[CLS]")
    start_label_ids.append(0)
    end_label_ids.append(0)
    segment_ids.append(0)
    for i,token in enumerate(raw_tokens):
        tokens.append(token)
        start_label_ids.append(start_labels[i])
        end_label_ids.append(end_labels[i])
        segment_ids.append(0)
    tokens.append("[SEP]")
    start_label_ids.append(0)
    end_label_ids.append(0)
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)


    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        start_label_ids.append(0)
        end_label_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(start_label_ids) == max_seq_length
    assert len(end_label_ids) == max_seq_length

    if ex_index < 5:
        logging.info("*** Example ***")
        logging.info("guid: %s" % example.guid)
        logging.info("tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens]))
        logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logging.info("start_labels: %s" % " ".join([str(x) for x in start_label_ids]))
        logging.info("end_labels: %s" % " ".join([str(x) for x in end_label_ids]))

    feature = InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            start_label_id=start_label_ids,
                            end_label_id=end_label_ids,
                            is_real_example=True)
    return feature


def file_based_convert_examples_to_features(examples, type2idx, max_seq_length, output_file, tokenizer):
    """Convert a set of `InputExample`s to a TFRecord file."""
    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, type2idx, max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["start_label_ids"] = create_int_feature(feature.start_label_id)
        features["end_label_ids"] = create_int_feature(feature.end_label_id)
        features["is_real_example"] = create_int_feature([int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {"input_ids": tf.FixedLenFeature([seq_length], tf.int64),
                        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
                        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
                        "start_label_ids": tf.FixedLenFeature([seq_length], tf.int64),
                        "end_label_ids": tf.FixedLenFeature([seq_length], tf.int64),
                        "is_real_example": tf.FixedLenFeature([], tf.int64)}

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, start_labels, end_labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(config=bert_config,
                               is_training=is_training,
                               input_ids=input_ids,
                               input_mask=input_mask,
                               token_type_ids=segment_ids,
                               use_one_hot_embeddings=use_one_hot_embeddings)
    # all_embedding=model.get_all_encoder_layers()
    # # 使用bert最后四层的输出
    # embedding=tf.reduce_mean(all_embedding[8:],axis=0)
    # dim=embedding.get_shape().as_list()[-1]
    # embedding=tf.layers.dense(embedding,units=dim)
    embedding = model.get_sequence_output()  # BERT模型输出的embedding  [batch_size,max_seq_len,embedding_size]
    # DGCNN
    dim = embedding.get_shape().as_list()[-1]
    embedding = tf.layers.conv1d(embedding, filters=dim, kernel_size=3, padding="same")
    embedding = tf.layers.conv1d(embedding, filters=dim, kernel_size=3, padding="same")
    # 做maxpooling，取序列长度上的最大值
    embedding -= (1.0 - tf.cast(tf.expand_dims(input_mask, 2), tf.float32)) * 1e10
    maxpooling=tf.reduce_max(embedding,axis=1) # [batch_size, embedding_size]

    # 将得到的maxpool拼接到embedding上，
    vec=tf.expand_dims(maxpooling,axis=1)
    vec=tf.zeros_like(embedding[:,:,:1])+vec
    output=tf.concat([embedding,vec],axis=2)

    # 一维卷积 relu激活
    output=tf.layers.conv1d(output,filters=128,kernel_size=3,activation=tf.nn.relu,padding="same") # [batch_size, max_seq_length, 128]

    # logits and loss
    start_logits=tf.layers.dense(output,units=num_labels) # [batch_size, max_seq_length, num_labels]
    end_logits=tf.layers.dense(output,units=num_labels) # [batch_size, max_seq_length, num_labels]
    
    start_loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=start_labels,logits=start_logits)
    end_loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=end_labels,logits=end_logits)
    start_loss=tf.reduce_sum(start_loss*tf.to_float(input_mask))/tf.reduce_sum(tf.to_float(input_mask))
    end_loss=tf.reduce_sum(end_loss*tf.to_float(input_mask))/tf.reduce_sum(tf.to_float(input_mask))
    loss=0.5*start_loss+0.5*end_loss
    return loss, start_logits, end_logits


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,num_train_steps, num_warmup_steps, use_tpu, use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        logging.info("*** Features ***")
        for name in sorted(features.keys()):
            logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        start_label_ids = features["start_label_ids"]
        end_label_ids = features["end_label_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        total_loss, start_logits,end_logits = create_model(bert_config=bert_config, 
                                                           is_training=is_training, 
                                                           input_ids=input_ids, 
                                                           input_mask=input_mask, 
                                                           segment_ids=segment_ids, 
                                                           start_labels=start_label_ids, 
                                                           end_labels=end_label_ids,
                                                           num_labels=num_labels, 
                                                           use_one_hot_embeddings=use_one_hot_embeddings)
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if init_checkpoint:
            assignment_map, initialized_variable_names = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            logging.info("  name = %s, shape = %s%s", var.name, var.shape,init_string)

        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            logging_hooks=tf.estimator.LoggingTensorHook({"loss":total_loss},every_n_iter=FLAGS.save_checkpoints_steps)
            output_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                     loss=total_loss,
                                                     train_op=train_op,
                                                     training_hooks=[logging_hooks])
        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_metrics={"eval_start_p:":tf_metrics.precision(start_label_ids,tf.argmax(tf.nn.softmax(start_logits,2),axis=2),num_labels),
                          "eval_start_r":tf_metrics.recall(start_label_ids,tf.argmax(tf.nn.softmax(start_logits,2),axis=2),num_labels),
                          "eval_start_f":tf_metrics.f1(start_label_ids,tf.argmax(tf.nn.softmax(start_logits,2),axis=2),num_labels),
                          "eval_end_p":tf_metrics.precision(end_label_ids,tf.argmax(tf.nn.softmax(end_logits,2),axis=2),num_labels),
                          "eval_end_r":tf_metrics.recall(end_label_ids,tf.argmax(tf.nn.softmax(end_logits,2),axis=2),num_labels),
                          "eval_end_f":tf_metrics.f1(end_label_ids,tf.argmax(tf.nn.softmax(end_logits,2),axis=2),num_labels),
                          }
            output_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                     loss=total_loss,
                                                     eval_metric_ops=eval_metrics)
        else:
            output_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                     predictions={"start_points": tf.nn.softmax(start_logits,axis=2),
                                                                  "end_points":tf.nn.softmax(end_logits,axis=2)})
        return output_spec

    return model_fn


def main(_):
    # tf.logging.set_verbosity(logging.DEBUG)

    #tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case, FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError("At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError("Cannot use sequence length {:d} because the BERT "
                         "model was only trained up to sequence length {:d}"
                         .format(FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    type2idx={"O":0,"Drug": 1,"Disease": 2,"Gene": 3,"ChemicalCompound": 4,"Virus": 5,"Chemical": 6,"Phenotype": 7,"Organization": 8}
    idx2type={val:key for key, val in type2idx.items()}
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    run_config = tf.estimator.RunConfig(model_dir=FLAGS.output_dir,
                                        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
                                        log_step_count_steps=1000)


    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = read_task_examples(input_file=FLAGS.train_file,is_training=True,tokenizer=tokenizer,type2idx=type2idx)
        num_train_steps = int(len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(bert_config=bert_config,
                                num_labels=len(type2idx),
                                init_checkpoint=FLAGS.init_checkpoint,
                                learning_rate=FLAGS.learning_rate,
                                num_train_steps=num_train_steps,
                                num_warmup_steps=num_warmup_steps,
                                use_tpu=False,
                                use_one_hot_embeddings=False)


    estimator = tf.estimator.Estimator(model_fn=model_fn,config=run_config,params={"batch_size":FLAGS.train_batch_size})

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        if not os.path.exists(train_file):
            file_based_convert_examples_to_features(train_examples, type2idx, FLAGS.max_seq_length, train_file, tokenizer)

        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(train_examples))
        logging.info("  Batch size = %d", FLAGS.train_batch_size)
        logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(input_file=train_file,
                                                     seq_length=FLAGS.max_seq_length,
                                                     is_training=True,
                                                     drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples = read_task_examples(FLAGS.dev_file,False,tokenizer,type2idx)
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        if not os.path.exists(eval_file):
            file_based_convert_examples_to_features(eval_examples, type2idx, FLAGS.max_seq_length, eval_file,tokenizer)
        logging.info("***** Running valid*****")
        logging.info("  Num examples = %d"%(len(eval_examples)))
        logging.info("  Batch size = %d", FLAGS.train_batch_size)
        eval_input_fn = file_based_input_fn_builder(input_file=eval_file,
                                                    seq_length=FLAGS.max_seq_length,
                                                    is_training=False,
                                                    drop_remainder=False)
        result=estimator.evaluate(input_fn=eval_input_fn,steps=len(eval_examples)//FLAGS.train_batch_size)
        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:
        predict_examples = read_task_examples(FLAGS.infer_file,False,tokenizer,type2idx)
        num_actual_predict_examples = len(predict_examples)

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        if not os.path.exists(predict_file):
            file_based_convert_examples_to_features(predict_examples, type2idx, FLAGS.max_seq_length, predict_file, tokenizer)

        logging.info("***** Running prediction*****")
        logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        logging.info("  Batch size = %d", FLAGS.predict_batch_size)


        predict_input_fn = file_based_input_fn_builder(input_file=predict_file,
                                                       seq_length=FLAGS.max_seq_length,
                                                       is_training=False,
                                                       drop_remainder=predict_drop_remainder)

        # 预测过程使用集成模型还是单个模型，若使用集成模型需要先生成，运行ensemble.py脚本
        ensemble_model_path=os.path.join(FLAGS.ensemble_dir,"checkpoint")
        if os.path.exists(ensemble_model_path):
            result = estimator.predict(input_fn=predict_input_fn,checkpoint_path=os.path.join(FLAGS.ensemble_dir,"average-0"))
        else:
            result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.txt")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            logging.info("***** Predict results *****")
            for (i, prediction) in enumerate(result):
                start_points=np.argmax(prediction["start_points"],axis=1)
                end_points=np.argmax(prediction["end_points"],axis=1)
                # mask
                doc_length=len(predict_examples[i].tokens)+2 # 因为加入了[CLS]和[SEP]两个符号，所以实际长度应该加2
                doc_tokens=" ".join(predict_examples[i].tokens)
                doc_start_labels=" ".join([idx2type[idx] for idx in start_points[:doc_length]])
                doc_end_labels=" ".join([idx2type[idx] for idx in end_points[:doc_length]])
                writer.write(doc_tokens+"\t"+doc_start_labels+"\t"+doc_end_labels+"\n")
                if i >= num_actual_predict_examples:
                    break



if __name__ == "__main__":
    flags.mark_flag_as_required("train_file")
    flags.mark_flag_as_required("dev_file")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()

