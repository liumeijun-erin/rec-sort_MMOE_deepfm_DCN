'''
model模块中模型训练部分：
DCN模型定义文件：配置文件; 定义结构+计算loss; setup_graph:建立图框架+梯度更新过程
Q1:cross结构实现 ：x0 传递；xi，xo迭代计算
Q2:如何实现层内w，b共享 -- 不能直接用dense + unit；需要tf计算公式实现
'''

import tensorflow as tf

config = {  
    'feature_len': 10,
    'embedding_dim': 5,
    'label_len': 1,
    'n_parse_threads': 4,
    'shuffle_buffer_size': 1024,
    'prefetch_buffer_size': 1,
    'batch_size': 16,

    'learning_rate': 0.01,
    'mlp_hidden_units' : [64,32],
    'cross_layer_num':2,
    'activation_function' : tf.nn.relu,
    'mlp_l2' : 0.1,

    'train_path': 'data/train',
    'test_path': 'data/val',
    'saved_embedding': 'data/saved_dcn_embedding',

    'max_steps': 80000,
    'train_log_iter': 1000,
    'test_show_iter': 1000,
    'last_test_auc':0.5,

    'saved_checkpoint': 'data/dcn/saved_ckpt',
    'checkpoint_name': 'dcn',
    'saved_pd':'data/dcn/saved_pd',

    "input_tensor": ["input_tensor"],
    "output_tensor": ["output_tensor"]
}

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

def cross_layers(variable_scope, inputs, cross_layer_num):
    '''
    input shape:[batch, feature_num * embedding_dimension] -- 已经凭借好连续feature+离散feature embed
    kernel_shape:[feature_num * embedding_dimension, 1]
    bias_shape:[feature_num * embedding_dimension, 1]
    '''
    with tf.variable_scope(variable_scope):
        input_shape = inputs.get_shape().as_list() 
        kernels = []
        bias = []
        for i in range(cross_layer_num):
            kernels.append(
                tf.get_variable('w_'+str(i), [input_shape[-1], 1],\
                    initializer = tf.glorot_normal_initializer(seed=0))

            )
            bias.append(
                tf.get_variable('b_'+str(i),[input_shape[-1], 1],\
                initializer = tf.zeros_initializer())
            )
        
        x_0 = tf.expand_dims(inputs, axis= 2)  # [batch, feature_num * embedding_dimension,1]
        x_i = x_0
        for i in range(cross_layer_num):
            # tf.tensordot作用: tf.matmul(x.T, y) ->-batch * all_len * 1 -> [batch*1*all_len] matmal[all_len*1]->[batch, 1, 1]
            x_j = tf.tensordot(x_i, kernels[i], axes=(1,0))  #[batch, 1, 1]
            dot_ = tf.matmul(x_0, x_j) # [batch * all_len * 1] matmul [batch, 1, 1] # 内置转置有匹配即可 -> batch *len * 1
            x_i = dot_ + x_j + bias[i] 
        x_i = tf.squeeze(x_j, axis = 2)
    return x_i

def dcn_fn(inputs, is_test):
    input_feature_emb = tf.reshape(inputs['feature_embedding'],\
        shape=[-1,config['feature_len'] * config['embedding_dim']])

    input_feature_emb = tf.reshape(input_feature_emb,
        [-1, config['feature_len'] * config['embedding_dim']])

    # mlp
    # v部分作为wembedding 平铺作为mlp部分的输入
    mlp_out_ = nn_tower(
        'dcn_hidden',
        input_feature_emb,
        config['mlp_hidden_units'],
        use_bias = True,
        l2 = config['mlp_l2'],
        activation = config['activation_function']
    )

    cross_out_ = cross_layers('cross', inputs=input_feature_emb, cross_layer_num=config['cross_layer_num'])
    out_ = tf.concat([cross_out_, mlp_out_], axis = 1)
    out_ = nn_tower('out', out_, [1], activation=None)
    out_tmp = tf.sigmoid(out_)

    if is_test:
        tf.add_to_collections('input_tensor', input_feature_emb)
        tf.add_to_collections('output_tensor', out_tmp)
    
    loss_ = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits = out_, labels=inputs['label']
    ))

    out_dic = {
        'loss': loss_,
        'ground_truth': inputs['label'][:, 0],
        'prediction': out_[:, 0]
    }
    return out_dic

def setup_graph(inputs, is_test=False):
    result = {}
    with tf.variable_scope('net_graph', reuse=is_test):

        net_out_dict = dcn_fn(inputs, is_test)

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

