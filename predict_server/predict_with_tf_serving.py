'''
部署方法2：docker + tf.serving在线部署
- docker ps -a
- docker images
- docker run -p 8501:8501 --mount type=bind,\
    source=xxx, target=/models/dcn -e MODEL_NAME=dcn \
    -t tensorflow/serving:1.10.0
    # 本地source文件运行在容器target位置; 加载镜像tensorflow/serving，注意版本一致
- 向url发出请求;数据格式如signature_def中定义
  可以用curl -d '{"instances": [{"x1": [1]}]}' -X POST http://localhost:9001/v1/models/lr:predict
  也可以使用requests库
'''
import json
import requests

saved_pb = "data/dcn/saved_pb"
feature_embeddings = "data/saved_dcn_embedding"
embedding_dim = 5
feature_len = 10
batch_size = 16


def load_embed():
    hashcode_dict = {}

    with open(feature_embeddings, "r") as lines:
        for line in lines:
            tmp = line.strip().split("\t")
            if len(tmp) == 2:
                vec = [float(_) for _ in tmp[1].split(",")]
                hashcode_dict[tmp[0]] = vec

    return hashcode_dict

def batch_predict(line_arr):
    """
    curl -d '{"instances": [{"x1": [1]}]}' -X POST http://localhost:9001/v1/models/lr:predict
    """
    embed_dict = load_embed()
    dim = embedding_dim
    input_size = feature_len * embedding_dim

    url = "http://172.30.31.8:8501/v1/models/dcn:predict"

    init_arr = [0.0] * input_size
    batch = []
    result = []
    for line in line_arr:
        tmp_arr = init_arr.copy()
        for i, f in enumerate(line):
            tmp = embed_dict.get(f, None)
            if tmp is not None:
                tmp_arr[i * dim:(i + 1) * dim] = tmp
        batch.append({"input0": tmp_arr})
        if len(batch) > batch_size:
            instances = {"instances": batch}

            json_response = requests.post(url, data=json.dumps(instances))
            res = json.loads(json_response.text)
            for p in res['predictions']:
                result.append(p[0])
            batch = []
    return result

if __name__ == '__main__':
    lines = open('data/hash_data/test_hash)
    line_arr = []
    for line in lines:
        line_arr.append(line.strip().split(","))
    print(len(line_arr))
    result = batch_predict(line_arr)
