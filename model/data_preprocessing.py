# -*- coding: utf-8 -*-

'''
model模块中数据流预处理部分:
* 读取数据，划分数据集-train val(labeled)+test(non-label)
* 定义哈希 + 将所有特征项进行hash(名+值);
* 转换为tfrecords类型保存+验证
'''
import os
import pandas as pd
import tensorflow as tf
import random

def bkdt2hash64(str01):
    mask60 = 0x0fffffffffffffff
    seed = 131
    hash = 0
    for s in str01:
        hash = hash * seed + ord(s)
    return hash & mask60

def tohash(data,column_names, save_path1, save_path2, split = False):
    wf1 = open(save_path1, 'w')
    if split:
        wf2 = open(save_path2, 'w')
    print(len(column_names))
    print(column_names)
    for line in data.iloc[:,:-1].values:
        line_hash = ''
        for i,col in enumerate(column_names):
            value_hash = bkdt2hash64(column_names[i]+str(line[i]))
            line_hash += str(value_hash) + ','
        if split:
            line_hash += str(line[-1]) 
        else :
            line_hash = line_hash.rstrip(',')
        line_hash += '\n'
        if split and random.randint(1,10) > 8:
            wf2.write(line_hash)
        else:
            wf1.write(line_hash)
            
    wf1.close()    
    if split:
        wf2.close()

def get_tfrecords_example(feature, label):
    tfrecords_features = {
        'feature': tf.train.Feature(int64_list=tf.train.Int64List(value=feature)),
        'label': tf.train.Feature(float_list=tf.train.FloatList(value=label))
    }
    return tf.train.Example(
        features=tf.train.Features(feature=tfrecords_features)
    )

def totfrecords(file, save_dir):
    print("Generating tfrecords from file: %s..." % file)
    file_num = 0
    writer = tf.python_io.TFRecordWriter(save_dir + '/' + 'part-%06d'%file_num + '.tfrecords')
    with open(file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            tmp = line.strip().split(',')
            feature = []
            for tmp_hash_value in tmp[:-1]:
                feature.append(int(tmp_hash_value))
            label = [float(tmp[-1])]
            example = get_tfrecords_example(feature, label)
            writer.write(example.SerializeToString())
            if (i+1) % 20000 == 0:
                writer.close()
                file_num += 1
                writer = tf.python_io.TFRecordWriter(save_dir + '/' + 'part-%06d' % file_num + '.tfrecords')
    writer.close()

def test_read_tfrecords(tfrecords_path,feature_len,label_len):
    tfrecord_path = os.path.join(tfrecords_path, os.listdir(tfrecords_path)[0])
    features = {
        "feature": tf.FixedLenFeature(feature_len, tf.int64),
        "label": tf.FixedLenFeature(label_len, tf.float32),
    }
    for serialized_example in tf.python_io.tf_record_iterator(tfrecord_path):
        example = tf.train.Example().ParseFromString(serialized_example)
        example = tf.parse_single_example(serialized_example, features)
        feature = example['feature']
        label = example['label']
        with tf.Session():  
            print(feature.eval())
            print(label.eval())
        break
    

if __name__ == '__main__':
    train_data_file = 'data/train.csv'
    test_data_file = 'data/test.csv'

    train_ds = pd.read_csv(train_data_file)
    test_ds = pd.read_csv(test_data_file)
    feature_names = test_ds.columns.values[2:]

    train_hash_path = 'data/hash_data/train_hash'
    val_hash_path = 'data/hash_data/val_hash'
    test_hash_path = 'data/hash_data/test_hash'
    if not os.path.exists('data/hash_data'):
        os.mkdir('data/hash_data')
        tohash(train_ds, feature_names, train_hash_path, val_hash_path, split = True)
        tohash(test_ds, feature_names, test_hash_path, test_hash_path, split = False)
        print("Hash process finished.")

    train_tfrecords_path = 'data/train'
    test_tfrecords_path = 'data/test'
    val_tfrecords_path = 'data/val'
    if not os.path.exists(train_tfrecords_path):
        os.mkdir(train_tfrecords_path)
        totfrecords(train_hash_path, train_tfrecords_path)
    print("train tfrecords files generated.")
    if not os.path.exists(val_tfrecords_path):
        os.mkdir(val_tfrecords_path)
        totfrecords(val_hash_path, val_tfrecords_path)
    print("val tfrecords files generated.")
    # if not os.path.exists(test_tfrecords_path):
    #     os.mkdir(test_tfrecords_path)
    #     totfrecords(test_hash_path, test_tfrecords_path)
    # print("test tfrecords files generated.")

    test_read_tfrecords(train_tfrecords_path,len(feature_names),1)