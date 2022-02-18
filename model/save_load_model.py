'''
model模块中模型训练部分：
model保存与加载:ckpt, pb两种类型文件
从ckpt生成的三种文件中生成pb文件(精简)
'''

import os
import tensorflow as tf
from tensorflow.python.framework import tensor_util
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.python.saved_model import tag_constants

def save_model_to_ckpt(sess=None, saver=None, save_path=None,
                       checkpoint_name=None, step=1):
    if save_path == "":
        print("save path is null")
        return
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, checkpoint_name)
    save_file = saver.save(sess, save_path, global_step=step)
    print("Net params saved to %s" % save_file)

# 从ckpt三种文件中生成pb文件
def _load_ckpt(model_path_dir):
    tf.reset_default_graph()
    ckpt = tf.train.get_checkpoint_state(model_path_dir)
    print("get optimum checkpoint : ", ckpt)
    saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
    session = tf.Session()
    saver.restore(session, ckpt.model_checkpoint_path)
    graph = tf.get_default_graph()
    return session, graph

def _compress_graph(sess, graph, input_op_name, output_op_name):
    graph_def = graph.as_graph_def()

    # 固化与output_op_name有关的所有节点，其他删除
    freeze_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        graph_def,
        output_op_name)

    # 清除节点上的device信息
    for node in freeze_graph_def.node:
        node.device = ""

    # 执行graph优化
    transforms = [
        'remove_nodes(op=Identity)',
        'merge_duplicate_nodes',
        'strip_unused_nodes',
        'fold_constants(ignore_errors=true,clear_output_shapes=true)',
        'fold_batch_norms',
    ]

    optimized_graph_def = TransformGraph(
        freeze_graph_def,
        input_op_name,
        output_op_name,
        transforms)

    # 重新调整输出shape的大小
    node_map = {}
    for node in freeze_graph_def.node:
        node_map[node.name] = node

    for node in freeze_graph_def.node:
        if node.op == 'Reshape':
            shape_node_name = node.input[1]
            shape_node = node_map[shape_node_name]
            if shape_node is not None and shape_node.op == 'Const':
                shape_array = tensor_util.MakeNdarray(
                    shape_node.attr['value'].tensor)
                if shape_array[0] > 1:
                    shape_array[0] = -1
                    new_shape_value = tf.make_tensor_proto(shape_array)
                    shape_node.attr[
                        'value'].tensor.tensor_content = new_shape_value.tensor_content

    return optimized_graph_def

def save_model_to_pb(model_path_dir, export_path_dir, input_tensor, output_tensor):
    # load model
    print("load checkpoint file to sess graph")
    session, graph = _load_ckpt(model_path_dir)
    print("model load success:")
    print(session.run(tf.report_uninitialized_variables()))
    # param input output
    input_op_name = []
    input_tensor_name = []
    print(graph.get_all_collection_keys())
    for name in input_tensor:
        input_op_name += [v.op.name for v in tf.get_collection(name)]
        input_tensor_name += [v.name for v in tf.get_collection(name)]

    assert (len(input_op_name) == 1)

    output_op_name = []
    output_tensor_name = []
    for name in output_tensor:
        output_op_name += [v.op.name for v in tf.get_collection(name)]
        output_tensor_name += [v.name for v in tf.get_collection(name)]
    # assert (len(output_op_name) == 1)

    inputs = {}
    for i, inp in enumerate(input_tensor_name):
        inputs['input' + str(i)] = graph.get_tensor_by_name(inp)
    outputs = {}
    for i, outp in enumerate(output_tensor_name):
        outputs['output' + str(i)] = graph.get_tensor_by_name(outp)
    print("===========", outputs)

    # optional：重置图，换上新的优化图
    # 如果去掉这部分得到得到更详细的模型图
    # 但是部署只需要简化的即可
    optimized_graph_def = _compress_graph(session, graph, input_op_name,
                                          output_op_name)
    tf.reset_default_graph()
    tf.import_graph_def(optimized_graph_def, name='')

    tf.saved_model.simple_save(
        session,
        export_path_dir,
        inputs=inputs,
        outputs=outputs)

#  tensorboard --logdir log
#  http://localhost:6006
def tensorboard_show_graph(model_dir):
    sess = tf.Session()
    meta_graph_def = tf.saved_model.loader.load(
        sess, [tag_constants.SERVING], model_dir)
    print(meta_graph_def)
    graph = tf.get_default_graph()
    summary_write = tf.summary.FileWriter('./log', graph)