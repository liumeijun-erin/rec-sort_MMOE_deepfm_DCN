'''
model模块中模型训练部分：
模型训练文件: 初始化参数服务，train/test 评估对象，定义模型网络（连接输入和图）；
写train函数-session.run；写valid函数;
保存模型为在线预估需要的pb格式
'''

from ps_fn import PS 
from input_fn import InputFn 
from auc_fn import AUCUtils
from dcn import setup_graph, config
# from deepfm import setup_graph, config
# from fnn import setup_graph, config
from save_load_model import save_model_to_ckpt, save_model_to_pb, tensorboard_show_graph
import tensorflow as tf
import os
print()

print(tf.__version__)
local_ps = PS(config['embedding_dim'])
train_metric = AUCUtils()
test_metric = AUCUtils()
inputs = InputFn(local_ps, config)

max_steps = config['max_steps']
train_log_iter = config['train_log_iter']
test_show_iter = config['test_show_iter']
last_test_auc = config['last_test_auc']

train_iter, train_inputs = inputs.input_fn(config['train_path'], is_test = False)
train_dic = setup_graph(train_inputs, is_test = False)
test_iter, test_inputs = inputs.input_fn(config['test_path'], is_test = True)
test_dic = setup_graph(test_inputs, is_test = True)

def train():
    _iter = 0
    print('#' * 80)
    saver = tf.train.Saver(max_to_keep=1)
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(),
                tf.local_variables_initializer()])
        sess.run(train_iter.initializer)
        while _iter < max_steps:
            old_embedding, new_embedding ,keys, out_, _ = sess.run([
                train_dic['feature_embedding'],
                train_dic['feature_new_embedding'],
                train_dic['feature'],
                train_dic['out'],
                train_dic['train_op']
            ])

            train_metric.add(
                out_['loss'],
                out_['ground_truth'],
                out_['prediction'])

            local_ps.push(keys, new_embedding)
            
            _iter += 1
            if _iter % train_log_iter == 0:
                print('Train at step %d: %s'% (_iter, train_metric.calc_str()))
                train_metric.reset()
            if _iter % test_show_iter == 0:
                valid_step(sess, test_iter, test_dic, saver, _iter)
            
def valid_step(sess, test_iter, test_dic, saver, _iter):
    test_metric.reset()
    sess.run(test_iter.initializer)
    global last_test_auc
    while True:
        try:
            out = sess.run(test_dic['out'])
            test_metric.add(
                out['loss'],
                out['ground_truth'],
                out['prediction']
            )
        except tf.errors.OutOfRangeError:
            print('Test at step%d:%s'% (_iter, test_metric.calc_str()))
            if test_metric.calc()['auc'] > last_test_auc:
                save_model_to_ckpt(
                    sess, saver, config['saved_checkpoint'],config['checkpoint_name'], _iter
                )
                last_test_auc = test_metric.calc()['auc'] 
                local_ps.save(config['saved_embedding'])
            break

def save_pb():
    input_tensor = config['input_tensor']
    output_tensor = config['output_tensor']
    model_path_dir = config['saved_checkpoint']
    export_path_model = config['saved_pd']
    save_model_to_pb(model_path_dir, export_path_model, input_tensor, output_tensor)


if __name__ == '__main__':
    train()
    save_pb()
    tensorboard_show_graph(config['saved_pd'])
    #  tensorboard --logdir log
    #  http://localhost:6006


        