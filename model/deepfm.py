'''
model模块中模型训练部分：
DeepFM模型定义文件：配置文件; 定义结构+计算loss; setup_graph:建立图框架+梯度更新过程
'''

import tensorflow as tf

config = {  
    'feature_len': 11,
    'embedding_dim': 8,
    'label_len': 2,
    'n_parse_threads': 4,
    'shuffle_buffer_size': 1024,
    'prefetch_buffer_size': 1,
    'batch_size': 32,
    'learning_rate': 0.01,

    'deepfm_hidden_units' : [64,32],
    'activation_function' : tf.nn.relu,
    'deepfm_l2' : 0.1,

    'train_path': 'data/train',
    'test_path': 'data/test',
    'saved_embedding': 'data/saved_deepfm_embedding',

    'max_steps': 30000,
    'train_log_iter': 3000,
    'test_show_iter': 3000,
    'last_test_auc':0.5,

    'saved_checkpoint': 'data/deepfm/saved_ckpt',
    'checkpoint_name': 'deepfm',
    'saved_pd':'data/deepfm/saved_pd',

    "input_tensor": ["input_tensor"],
    "output_tensor": ["output_tensor"]
}

# TODO: pretrained-先用FM训练结果进行结果初始化 -- 修改PS初始化方法
def nn_tower(name, nn_input, hidden_units, activation=tf.nn.relu, use_bias=False, l2=0.0):
    out = nn_input
    for i, num in enumerate(hidden_units):
        out = tf.layers.dense(
            out,
            units = num,
            kernel_initializer = tf.contrib.layers.xavier_initializer(),
            kernel_regularizer = tf.contrib.layers.l2_regularizer(l2),
            use_bias = use_bias,
            activation=activation,
            name= name+'/layer_' + str(i)
        )
    return out

def deepfm_fn(inputs, is_test):
    input = tf.reshape(inputs['feature_embedding'],\
        shape=[-1,config['feature_len'], config['embedding_dim']])
    
    # 将embedding划分为w+v;最后一维作为线性权重w
    input_ = tf.split(
        input,
        num_or_size_splits=[config['embedding_dim']-1, 1],
        axis = 2
    )
    # fm
    fm_bias = tf.get_variable('bias',[1,],initializer=tf.zeros_initializer())  # b
    fm_linear = tf.nn.bias_add(tf.reduce_sum(input_[1],axis=1),fm_bias)  # sum(wx)
    square_sum = tf.reduce_sum(tf.square(input_[0]), axis=1)
    sum_square = tf.square(tf.reduce_sum(input_[0],axis=1))
    fm_cross = (sum_square - square_sum) / 2.0


    # mlp
    # v部分作为wembedding 平铺作为mlp部分的输入
    feature_embedding = tf.reshape(input_[0], [-1, config['feature_len']*(config['embedding_dim'] - 1)])
    mlp_out_ = nn_tower(
        'deepfm_hidden',
        feature_embedding,
        config['deepfm_hidden_units'],
        use_bias = True,
        l2 = config['deepfm_l2'],
        activation = config['activation_function']
    )

    out_ = tf.concat([fm_linear, fm_cross, mlp_out_], axis = 1)
    out_ = nn_tower('out', out_, [1], activation=None)
    pctcvr = tf.reshape(out_,[-1])
    out_tmp = tf.sigmoid(out_)

    if is_test:
        tf.add_to_collections('input_tensor', input)
        tf.add_to_collections('output_tensor', out_tmp)
    
    loss_ = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits = pctcvr, labels=inputs['label'][:,1]
    ))

    out_dic = {
        'loss': loss_,
        'ground_truth': inputs['label'][:, 1],
        'prediction': pctcvr
    }
    return out_dic

def setup_graph(inputs, is_test=False):
    result = {}
    with tf.variable_scope('net_graph', reuse=is_test):
        net_out_dict = deepfm_fn(inputs, is_test)
        loss = net_out_dict['loss']
        result['out'] = net_out_dict
        if is_test:
            return result
        
        # 对embedding梯度下降
        emb_grad = tf.gradients(loss,[inputs['feature_embedding']], name='feature_embedding')[0]
        result['feature_new_embedding'] = \
            inputs['feature_embedding'] - config['learning_rate'] * emb_grad
        
        result['feature_embedding'] = inputs['feature_embedding']
        result['feature'] = inputs['feature']

        # 对fnn中mlp参数也梯度下降
        tvars1 = tf.trainable_variables()
        grads1 = tf.gradients(loss, tvars1)
        opt = tf.train.GradientDescentOptimizer(
            learning_rate = config['learning_rate'],
            use_locking = True
        )
        train_op = opt.apply_gradients(zip(grads1, tvars1))
        result['train_op'] = train_op

        return result