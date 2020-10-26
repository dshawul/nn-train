from __future__ import print_function
import os
import argparse
import struct
from train import my_load_model

#import tensorflow and set logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except:
    pass

SAVE_BIN = True
VERSION = 0

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

def copy_weights(m1,m2):
    """
    Copy weights from one model to another
    """
    for layer1,layer2 in zip(m1.layers,m2.layers):
        w1 = layer1.get_weights()
        if(len(w1) > 0):
            w2 = layer2.get_weights()
            for k in range(len(w2)):
                w2[k] = w1[k].reshape(w2[k].shape)
            layer2.set_weights(w2)

def save_weights(m,name):
    """
    Save weights in raw format
    """
    myF = open(name,"wb")
    data = struct.pack('i',VERSION)
    myF.write(data)
    for layer in m.layers:
        w = layer.get_weights()
        if(len(w) > 0):
            for wi in w:
                print(wi.shape)
                wf = wi.flatten()
                for x in wf:
                    data = struct.pack('f',x)
                    myF.write(data)

def convertGraph(modelPath, inferPath, outdir, name):

    from tensorflow.python.framework import graph_io

    tf.keras.backend.set_learning_phase(0)
    sess = tf.keras.backend.get_session()

    #defaults
    if outdir == None:
        outdir = os.path.dirname(modelPath)
    if name == None:
        name = os.path.basename(modelPath)

    #create dir
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    if inferPath != None:
        #copy weights
        opt_model = my_load_model(modelPath)
        net_model = my_load_model(inferPath)
        copy_weights(opt_model, net_model)

        #save and reload inference model with cleared session
        modelPath = os.path.join(outdir, "_temp_")
        net_model.save(modelPath, include_optimizer=False, save_format='h5')
        tf.keras.backend.clear_session()
        tf.keras.backend.set_learning_phase(0)
        sess = tf.keras.backend.get_session()
        net_model = my_load_model(modelPath)
    else:
        net_model = my_load_model(modelPath)

    # My net format
    if SAVE_BIN:
        fpath = os.path.join(outdir, name + ".bin")
        save_weights(net_model, fpath)
        print('Raw weights: ', fpath)

    #freeze graph
    output_names = [out.op.name for out in net_model.outputs]
    constant_graph = freeze_session(sess, output_names=output_names)

    # Write the graph in binary .pb file
    graph_io.write_graph(constant_graph, outdir, name + ".pb", as_text=False)
    print('Frozen graph: ', os.path.join(outdir, name + ".pb"))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model','-m', dest='model', required=True, \
        help='REQUIRED: The HDF5 Keras model you wish to convert to .pb')
    parser.add_argument('--infer-model','-i', dest='infer_model', required=False, \
        help='Inference model on which to copy weights to')
    parser.add_argument('--outdir','-o', dest='outdir', required=False, default=None, \
        help='The directory to place the output files')
    parser.add_argument('--name','-n', dest='name', required=False, default=None, \
        help='The name of the resulting output graph')
    args = parser.parse_args()

    convertGraph( args.model,  args.infer_model, args.outdir, args.name)
