'''
model模块中模型训练部分：
MMOE模型定义文件：配置文件; 定义结构+计算loss; setup_graph:建立图框架+梯度更新过程
Q1:各个特征提取器--都用相同mlp
Q2:loss如何定义:loss 加和 但是要保证两者之间相关；否则联合求导无意义:pctcvr 由ctr算出
Q3: 两个任务如何联系在一起/意义：
两个任务 ctr cvr：在cvr上下效果更好。
只从点击样本学习cvr -> 样本有偏; 从全样本空间学习cvr -> 样本过于稀疏，emebdding 训练不充分
加入ctr帮助训练得更充分；且ctcvr/ctr == cvr除偏
从取
Q4:embedding vs 专家块 -- 注意不是替代关系，专家块为一个mlp保存参数即可
'''

from ast import expr_context
import tensorflow as tf

config = {  
    'feature_len': 11,
    'embedding_dim': 8,
    'label_len': 2,
    'n_parse_threads': 4,
    'shuffle_buffer_size': 1024,
    'prefetch_buffer_size': 1,
    'batch_size': 32,
    'expert_num':  3,
    'target_num': 2,
    'learning_rate': 0.01,
    'mlp_hidden_units' : [32, 16],
    'activation_function' : tf.nn.relu,
    'mlp_l2' : 0.1,
    'loss_weight':[0.5, 0.5],

    'train_path': 'data/train',
    'test_path': 'data/test',
    'saved_embedding': 'data/saved_dnn_embedding',

    'max_steps': 30000,
    'train_log_iter': 300,
    'test_show_iter': 300,
    'last_test_auc':0.5,

    'saved_checkpoint': 'data/mmoe/saved_ckpt',
    'checkpoint_name': 'mmoe',
    'saved_pd':'data/mmoe/saved_pd',

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

def mmoe_fn(inputs, is_test):
    input_feature_emb = tf.reshape(inputs['feature_embedding'],\
        shape=[-1,config['feature_len'] * config['embedding_dim']])

    # experts 部分
    # 双层
    with tf.variable_scope('experts'):
        experts = []
        for i in range(config['expert_num']):
            expert_out = nn_tower(
                'expert_mlp_hidden_'+str(i),input_feature_emb, config['mlp_hidden_units'],
                use_bias=True, activation=config['activation_function'],l2=config['mlp_l2']
            )
            experts.append(expert_out)  # [expert_num, batch, expert_out_dim]
        experts = tf.transpose(tf.stack(experts, axis = 0), perm=[1,0,2])  # [expert_num, batch, expert_out_dim] -> [batch, expert_num, expert_out_dim]

    # 门控 部分
    # 单层网络
    with tf.variable_scope('gates'):
        gates = []
        for i in range(config['target_num']):
            gate = nn_tower(
                'gate_mlp_hidden_'+str(i), input_feature_emb, [config['expert_num']],
                use_bias=True, activation=config['activation_function'],l2=config['mlp_l2']
            )
            gates.append(tf.expand_dims(tf.nn.softmax(gate),1))  # list([batch,1,expert_num]*target_num)

    # 双塔模型输入部分
    # 加权求和即可
    with tf.variable_scope('tower_input'):
        tower_inputs = []
        for i in range(config['target_num']):
            # gates[i] -  [batch,1,expert_num]
            # experts - [batch, expert_num, expert_out_dim]
            tower_inputs.append(tf.reshape(tf.matmul(gates[i], experts), shape=[-1,config['mlp_hidden_units'][-1]]))
            # [batch, expert_out_dim]
    
    # 双塔模型部分
    # 双层
    tower_outs = []
    for i in range(config['target_num']):
        with tf.variable_scope('tower_'+str(i)):
            tower_out = nn_tower(
                'tower_mlp_hidden_'+str(i),tower_inputs[i], config['mlp_hidden_units'],
                use_bias=True, activation=config['activation_function'],l2=config['mlp_l2']
            )
            tower_out = nn_tower(
                'tower_out_'+str(i),tower_inputs[i], [1],
                use_bias=True, activation=None, l2=config['mlp_l2']
            )
            tower_outs.append(tower_out)

    # 总output
    with tf.variable_scope('mmoe_out'):
        pctr = tower_outs[0]
        pctcvr = tower_outs[0] * tower_outs[1]

    pctr = tf.reshape(pctr,[-1])
    pctcvr = tf.reshape(pctcvr,[-1])
    pred = tf.sigmoid(tower_outs)

    if is_test:
        tf.add_to_collections('input_tensor', input_feature_emb)
        tf.add_to_collections('output_tensor', pred)
    
    loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits = pctr, labels=inputs['label'][:,0]
    ))

    loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits = pctcvr, labels=inputs['label'][:,1]
    ))

    loss_ = config['loss_weight'][0]* loss1 + config['loss_weight'][1] * loss2

    out_dic = {
        'loss': loss_,
        'ground_truth': inputs['label'][:, 0],
        'prediction': pctr
    }
    return out_dic

def setup_graph(inputs, is_test=False):
    result = {}
    with tf.variable_scope('net_graph', reuse=is_test):

        net_out_dict = mmoe_fn(inputs, is_test)

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

