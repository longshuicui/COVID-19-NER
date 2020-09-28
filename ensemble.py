# -*- coding: utf-8 -*-
"""
@author: longshuicui
@date  :   2020/9/28
@function: single model ensemble
"""
import os
import numpy as np
import argparse
import tensorflow as tf
from collections import OrderedDict

#
# flags=tf.flags
# FLAGS=flags.FLAGS
# flags.DEFINE_string("dev_file", "task1_public/new_dev.json","The dev file")
# flags.DEFINE_string("bert_config_file", "uncased_L-12_H-768_A-12/bert_config.json", "The config json file")
# flags.DEFINE_string("vocab_file", "uncased_L-12_H-768_A-12/vocab.txt", "The vocabulary file that the BERT model was trained on.")
# flags.DEFINE_string("output_dir", "./best_model","The output directory where the model checkpoints will be written.")
# flags.DEFINE_integer("max_seq_length", 512, "The maximum total input sequence length")
# flags.DEFINE_bool("do_lower_case", True, "Whether to lower case the input text.")
# flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")
# flags.DEFINE_integer("num_labels", 8, "Total batch size for predict.")
#
#
# def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, start_labels, end_labels, num_labels,
#                  use_one_hot_embeddings):
#     """Creates a classification model."""
#     model = modeling.BertModel(config=bert_config,
#                                is_training=is_training,
#                                input_ids=input_ids,
#                                input_mask=input_mask,
#                                token_type_ids=segment_ids,
#                                use_one_hot_embeddings=use_one_hot_embeddings)
#
#     embedding = model.get_sequence_output()  # BERT模型输出的embedding  [batch_size,max_seq_len,embedding_size]
#     # 做maxpooling，取序列长度上的最大值
#     embedding -= (1.0 - tf.cast(tf.expand_dims(input_mask, 2), tf.float32)) * 1e10
#     maxpooling = tf.reduce_max(embedding, axis=1)  # [batch_size, embedding_size]
#
#     # 将得到的maxpool拼接到embedding上，
#     vec = tf.expand_dims(maxpooling, axis=1)
#     vec = tf.zeros_like(embedding[:, :, :1]) + vec
#     output = tf.concat([embedding, vec], axis=2)
#
#     # 一维卷积 relu激活
#     output = tf.layers.conv1d(output, filters=128, kernel_size=3, activation=tf.nn.relu, padding="same")  # [batch_size, max_seq_length, 128]
#
#     # logits and loss
#     start_logits = tf.layers.dense(output, units=num_labels)  # [batch_size, max_seq_length, num_labels]
#     end_logits = tf.layers.dense(output, units=num_labels)  # [batch_size, max_seq_length, num_labels]
#
#     start_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=start_labels, logits=start_logits)
#     end_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=end_labels, logits=end_logits)
#     start_loss = tf.reduce_sum(start_loss * tf.to_float(input_mask)) / tf.reduce_sum(tf.to_float(input_mask))
#     end_loss = tf.reduce_sum(end_loss * tf.to_float(input_mask)) / tf.reduce_sum(tf.to_float(input_mask))
#     loss = 0.5 * start_loss + 0.5 * end_loss
#     return loss, start_logits, end_logits
#
#
# def infer():
#     # preprocess
#     type2idx = {"O": 0, "Drug": 1, "Disease": 2, "Gene": 3, "ChemicalCompound": 4, "Virus": 5, "Chemical": 6,
#                 "Phenotype": 7, "Organization": 8}
#     idx2type = {val: key for key, val in type2idx.items()}
#     bert_config=modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
#
#
#     ## model
#     input_ids=tf.placeholder(dtype=tf.int32,shape=[None,FLAGS.max_seq_length],name="input_ids")
#     input_mask=tf.placeholder(dtype=tf.int32,shape=[None,FLAGS.max_seq_length],name="input_mask")
#     segment_ids=tf.placeholder(dtype=tf.int32,shape=[None,FLAGS.max_seq_length],name="segment_ids")
#     start_labels=tf.placeholder(dtype=tf.int32,shape=[None,FLAGS.max_seq_length],name="start_labels")
#     end_labels=tf.placeholder(dtype=tf.int32,shape=[None,FLAGS.max_seq_length],name="end_labels")
#
#     model=create_model(bert_config=bert_config,
#                        is_training=False,
#                        input_ids=input_ids,
#                        input_mask=input_mask,
#                        segment_ids=segment_ids,
#                        start_labels=start_labels,
#                        end_labels=end_labels,
#                        num_labels=len(type2idx),
#                        use_one_hot_embeddings=False)
#
#     with tf.Session() as sess:
#         saver=tf.train.Saver(sess)
#         ckpt=tf.train.latest_checkpoint(checkpoint_dir=FLAGS.output_dir)
#         print(ckpt)
#         saver.restore(sess=sess,save_path=ckpt)

def parseargs():
    msg = "Average checkpoints"
    usage = "average.py [<args>] [-h | --help]"
    parser = argparse.ArgumentParser(description=msg, usage=usage)

    parser.add_argument("--path", type=str, default="./best_model", help="checkpoint dir")
    parser.add_argument("--checkpoints", type=int, default=5, help="number of checkpoints to use")
    parser.add_argument("--output", type=str, help="output path", default="./ensemble_model")

    return parser.parse_args()

def get_checkpoinsts(path):
    """获取ckpt模型的名称"""
    if not tf.gfile.Exists(os.path.join(path,"checkpoint")):
        raise ValueError("There is not checkpoint in %s"%path)
    checkpoint_names=[]
    with tf.gfile.GFile(os.path.join(path,"checkpoint")) as fd:
        fd.readline() # skip the first line
        for line in fd:
            name=line.strip().split(":")[-1].strip()[1:-1]
            key = int(name.split("-")[-1])
            checkpoint_names.append((key,os.path.join(path,name)))

    sorted_name=sorted(checkpoint_names,key=lambda x:x[0],reverse=True)

    return [item[-1] for item in sorted_name]

def checkpoint_exists(path):
    return tf.gfile.Exists(path) or tf.gfile.Exists(path+".meta") or tf.gfile.Exists(path+".index")


if __name__ == '__main__':
    args=parseargs()
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    tf.logging.set_verbosity(tf.logging.INFO)
    checkpoints=get_checkpoinsts(args.path)
    checkpoints=checkpoints[:args.checkpoints]
    checkpoints=[c for c in checkpoints if checkpoint_exists(c)]
    if not checkpoints:
        raise ValueError("None of checkpoints exist")
    var_list=tf.train.list_variables(checkpoints[0])
    var_values, var_dtypes=OrderedDict(),OrderedDict()
    for name, shape in var_list:
        if not name.startswith("global_step"):
            var_values[name]=np.zeros(shape)

    for checkpoint in checkpoints:
        tf.logging.info("Read from checkpoint %s", checkpoint)
        reader=tf.train.load_checkpoint(checkpoint)
        for name in var_values:
            tensor=reader.get_tensor(name)
            var_dtypes[name]=tensor.dtype
            var_values[name]+=tensor

    # average ckpt
    for name in var_values:
        var_values[name] /= len(checkpoints)

    tvars=[tf.get_variable(name,shape=var_values[name].shape,dtype=var_dtypes[name]) for name in var_values]
    placeholders=[tf.placeholder(v.dtype,shape=v.shape) for v in tvars]
    assign_op=[tf.assign(v,p) for v,p in zip(tvars,placeholders)]
    global_step=tf.Variable(0,name="global_step",trainable=False,dtype=tf.int64)
    saver=tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for p,assign_op,(name, value) in zip(placeholders,assign_op,var_values.items()):
            sess.run(assign_op,feed_dict={p:value})

        saved_name=os.path.join(args.output,"average")
        saver.save(sess,saved_name,global_step=global_step)

    tf.logging.info("Averaged Checkpoints saved in %s"%saved_name)
