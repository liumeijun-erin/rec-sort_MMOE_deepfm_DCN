# -*- coding: utf-8 -*-
'''
model模块中模型训练部分：
输入数据预处理文件 - 定义InputFn类： 读取tfrecords文件转化为tf.dataset统一格式，分批等
'''

import tensorflow as tf
import os

class InputFn:
    def __init__(self, local_ps, config):
        self.feature_len = config['feature_len']
        self.label_len =  config['label_len']
        self.n_parse_threads = config['n_parse_threads']
        self.shuffle_buffer_size = config['shuffle_buffer_size']
        self.prefetch_buffer_size = config['prefetch_buffer_size']
        self.batch = config['batch_size']
        self.local_ps = local_ps

    # 读取、解析tfrecords文件；dataset构建、分批+拉取ps中隐向量；返回迭代器
    def input_fn(self, data_dir, is_test = False):
        # 解析单条tfrecords数据
        def _parse_example(example):
            features = {
                "feature": tf.FixedLenFeature(self.feature_len, tf.int64),
                "label": tf.FixedLenFeature(self.label_len, tf.float32),
            }
            return tf.parse_single_example(example, features)

        # _get_weight代替_get_embedding
        def _get_embedding(parsed):
            keys = parsed["feature"]
            keys_array = tf.py_func(
                self.local_ps.pull,[keys],tf.float32
            )
            result = {
                'feature': parsed['feature'],
                'label': parsed['label'],
                'feature_embedding': keys_array
            }
            return result
        
        file_list = os.listdir(data_dir)
        files = []
        for i in range(len(file_list)):
            files.append(os.path.join(data_dir, file_list[i]))

        dataset = tf.data.Dataset.list_files(files)

        if is_test:
            dataset = dataset.repeat(1)
        else:
            dataset = dataset.repeat()

        dataset = dataset.interleave(
            lambda _: tf.data.TFRecordDataset(_),
            cycle_length = 1
        )

        dataset = dataset.map(
            _parse_example,
            num_parallel_calls = self.n_parse_threads
        )

        # 分批
        dataset = dataset.batch(self.batch, drop_remainder = True)

        # 在分batch之后对于每个batch请求参数服务器,因为参数在不断改变
        dataset = dataset.map(_get_embedding, num_parallel_calls = self.n_parse_threads)

        if not is_test:
            dataset.shuffle(self.shuffle_buffer_size)

        dataset = dataset.prefetch(buffer_size = self.prefetch_buffer_size)

        # 返回迭代器
        iterator = dataset.make_initializable_iterator()
        return iterator, iterator.get_next()

# if __name__ == '__main__':
#     from ps import PS
#     local_ps = PS(8)
#     inputs = InputFn(local_ps)
#     data_dir = "../data/train"
#     train_itor, train_inputs = inputs.input_fn(data_dir, is_test=False)
#     with tf.Session() as sess:
#         sess.run(train_itor.initializer)
#         for i in range(1):
#             print(sess.run(train_inputs))