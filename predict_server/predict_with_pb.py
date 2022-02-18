'''
预估服务部分：
从pd中加载模型；load_model return sess, input_tensor, output_tensor
请求特征服务器获得input；
基于特征值去参数服务器获取参数，如embedding(note:也在不断更新，不同于vector_server中)
输入模型，返回结果
'''
import random 
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
import numpy as np

batch_size = 16
feature_len = 10
embedding_dim = 5
saved_pd = 'data/dcn/saved_pd'
feature_embeddings = 'data/saved_dcn_embedding'

def load_model():
    sess = tf.Session()
    meta_graph_def = tf.saved_model.loader.load(sess,[tag_constants.SERVING],saved_pd)
    
    signature = meta_graph_def.signature_def
    in_tensor_name = signature[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['input0'].name
    out_tensor_name = signature[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['output0'].name

    in_tensor = sess.graph.get_tensor_by_name(in_tensor_name)
    out_tensor = sess.graph.get_tensor_by_name(out_tensor_name)

    return sess, in_tensor, out_tensor

def load_emb():
    hashcode_dict = {}
    with open(feature_embeddings, 'r') as lines:
        for line in lines:
            tmp = line.strip().split('\t')
            if len(tmp) == 2:
                vec = [float(_) for _ in tmp[1].split(',')]
                hashcode_dict[tmp[0]] = vec
    return hashcode_dict

# def get_request():
#     # 模拟实际，抽取request user+id 对
#     embed_dict = load_emb()
#     keys = embed_dict.keys()
#     hashcode = []
#     batch_sample = []
#     for _ in range(batch_size):
#         feature_value = random.sample(keys,8)
#         hashcode.append(','.join([str(_) for _ in feature_value]))  # 一个ui对
#         hashcode_embed = []
#         for s in feature_value:
#             hashcode_embed.append(embed_dict[s])
#         batch_sample.append(hashcode_embed)
#     return hashcode, batch_sample

def get_request_from_file_path(hash_file):
    sess, in_tensor, out_tensor = load_model()
    embed_dict = load_emb()   
    dim = embedding_dim
    input_size = feature_len * dim
    empty_line = [0.0] * input_size
    batch = []
    predict = []
    with open(hash_file, 'r') as f:
        for line in f.readlines():
            hash_line = line.strip().split(",")
            tmp_line = empty_line
            for i, value in enumerate(hash_line):
                tmp_embed = embed_dict.get(value, None)
                if tmp_embed is not None:
                    tmp_line[i*dim:(i+1)*dim] = tmp_embed
            # print(tmp_line)
            batch.append(tmp_line)
            if len(batch) >= batch_size:
                # print(batch)
                prediction = sess.run(out_tensor, feed_dict={in_tensor: np.array(batch)})
                for p in prediction:
                    predict.append(p[0])
                batch = []
    print(predict)
    return predict

if __name__=='__main__':        
    # requests, batch_sample = get_request()
    # sess, in_tensor, out_tensor = load_model()
    # predict = sess.run(out_tensor,feed_dict = {in_tensor: np.array(batch_sample)})
    predict = get_request_from_file_path('data/hash_data/test_hash')