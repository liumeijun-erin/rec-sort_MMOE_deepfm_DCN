# -*- coding: utf-8 -*-

'''
model模块中数据流预处理部分:
* 读取json数据
* 划分数据集-train test
* 定义哈希 + 将所有特征项进行hash(名+值);
* 转换为tfrecords类型保存+验证
'''
import os
import pandas as pd
import tensorflow as tf
import random
import json

def bkdt2hash64(str01):
    mask60 = 0x0fffffffffffffff
    seed = 131
    hash = 0
    for s in str01:
        hash = hash * seed + ord(s)
    return hash & mask60

def tohash_line(json_line):
    '''所选features
    ['user_id', 'item_id', 'sdk_type', 'remote_host', 'device_type',
    'dtu', 'click_goods_num', 'buy_click_num', 'goods_show_num',
    'goods_click_num', 'brand_name']'''

    features_selected_set = ['sdk_type', 'remote_host', 'device_type',
    'dtu', 'click_goods_num', 'buy_click_num', 'goods_show_num',
    'goods_click_num', 'brand_name']  
    feature_selected_type = ['StringValue','StringValue','StringValue',\
        'StringValue','FloatValue','FloatValue','FloatValue','FloatValue',\
        'StringValue'   ]
    feature_selected = dict(zip(features_selected_set,feature_selected_type))
    
    hash_line = []
    line_json = json.loads(json_line)   
    # print(line_json.keys()  # dict_keys(['UserId', 'ItemId', 'Label', 'UserFeature', 'ItemFeature'])
    # for key in line_json.keys():
    #     print(line_json[key])
    user_id = line_json['UserId']
    hash_line.append(str(bkdt2hash64("user_id=" + str(user_id))))
    item_id = line_json['ItemId']
    hash_line.append(str(bkdt2hash64("item_id=" + str(item_id))))
    label = line_json["Label"].split("$#") # click/conversion

    user_feature_dict_list = line_json['UserFeature'].get('Feature',None)
    item_feature_dict_list = line_json['ItemFeature'].get('Feature',None)
    if not user_feature_dict_list or not item_feature_dict_list:
        return
    user_feature_dict_new = {}
    item_feature_dict_new = {}
    for user_feature_dict in user_feature_dict_list:
       user_feature_dict_new[user_feature_dict['FeatureName']] = user_feature_dict['FeatureValue']
    for item_feature_dict in item_feature_dict_list:
       item_feature_dict_new[item_feature_dict['FeatureName']] = item_feature_dict['FeatureValue']
    # print(user_feature_dict_new.keys())
    # dict_keys(['device_id', 'user_id', 'sdk_type', 'remote_host', 'device_brand', 'device_type', 'dtu', 'click_goods_num', 'click_goods', 'click_goods_list', 'agm_page_num', 'detail_page_num', 'share_num', 'share_goods', 'share_goods_list', 'buy_click_num', 'buy_click_goods', 'buy_click_goods_list'])
    # print(item_feature_dict_new.keys())
    # dict_keys(['goods_buy_num', 'goods_click_num', 'goods_share_num', 'goods_show_num', 'brand_name', 'cate1', 'cate2', 'cate3', 'commission', 'goods_id', 'ist_date', 'lowest_price', 'price', 'site_id', 'source', 'status'])
    
    for user_feature_key, user_feature_value in user_feature_dict_new.items():
        if user_feature_key in feature_selected.keys():
            user_feature_value = user_feature_value[feature_selected[user_feature_key]]
            hash_line.append(str(bkdt2hash64(user_feature_key+"=" + str(user_feature_value))))
            # print(user_feature_key, user_feature_value)

    for item_feature_key, item_feature_value in item_feature_dict_new.items():
        if item_feature_key in feature_selected.keys():
            item_feature_value = item_feature_value[feature_selected[item_feature_key]]
            hash_line.append(str(bkdt2hash64(item_feature_key+'+' + str(item_feature_value))))

    hash_line.append(str(label[0]) + "," + str(label[1]))
    return ",".join(hash_line)
           
def extract_file_tohash(rfile_path, wfile_path):
    rfile = open(rfile_path, encoding='utf8')
    wfile = open(wfile_path, 'w')
    for line in rfile:
        tmp = tohash_line(line)
        if tmp is not None:
            wfile.write(tmp + '\n')
    wfile.close()

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
            for tmp_hash_value in tmp[:-2]:
                feature.append(int(tmp_hash_value))
            label = [float(tmp[-2]),float(tmp[-1])]
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
    train_file_path = 'data/170'
    test_file_path = 'data/180'
    hash_dir = 'data/hash_data'
    train_hash_path = 'data/hash_data/train_hash'
    test_hash_path = 'data/hash_data/test_hash'

    if not os.path.exists(hash_dir):
        os.mkdir(hash_dir)
        extract_file_tohash(train_file_path, train_hash_path)
        extract_file_tohash(test_file_path, test_hash_path)
        print("Hash process finished.")

    train_tfrecords_path = 'data/train'
    test_tfrecords_path = 'data/test'
    if not os.path.exists(train_tfrecords_path):
        os.mkdir(train_tfrecords_path)
        totfrecords(train_hash_path, train_tfrecords_path)
    print("train tfrecords files generated.")
    if not os.path.exists(test_tfrecords_path):
        os.mkdir(test_tfrecords_path)
        totfrecords(test_hash_path, test_tfrecords_path)
    print("test tfrecords files generated.")

    test_read_tfrecords(train_tfrecords_path,11,2)