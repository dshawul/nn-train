'''
This script converts a .h5 Keras model into a Tensorflow .pb file.

Attribution: This script was adapted from https://github.com/amir-abdi/keras_to_tensorflow

MIT License

Copyright (c) 2017 bitbionic

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import os
import argparse

try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except ImportError:
    import tensorflow as tf

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
    for layer1,layer2 in zip(m1.layers,m2.layers):
        w1 = layer1.get_weights()
        w2 = layer2.get_weights()
        for k in range(len(w2)):
            w2[k] = w1[k].reshape(w2[k].shape)
        layer2.set_weights(w2)

def convertGraph( modelPath, outdir, numoutputs, prefix, name, inferPath):
    '''
    Converts an HD5F file to a .pb file for use with Tensorflow.

    Args:
        modelPath (str): path to the .h5 file
           outdir (str): path to the output directory
       numoutputs (int):   
           prefix (str): the prefix of the output aliasing
             name (str):
    Returns:
        None
    '''

    from tensorflow.python.framework import graph_io

    tf.keras.backend.set_learning_phase(0)
    sess = tf.keras.backend.get_session()

    #create dir
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    if inferPath != None:
        #copy weights
        opt_model = tf.keras.models.load_model(modelPath)
        net_model = tf.keras.models.load_model(inferPath)
        copy_weights(opt_model, net_model)

        #save and reload inference model with cleared session
        modelPath = os.path.join(outdir, "_temp_")
        net_model.save(modelPath, include_optimizer=False, save_format='h5')
        tf.keras.backend.clear_session()
        tf.keras.backend.set_learning_phase(0)
        sess = tf.keras.backend.get_session()
        net_model = tf.keras.models.load_model(modelPath)
    else:
        net_model = tf.keras.models.load_model(modelPath)

    # Alias the outputs in the model - this sometimes makes them easier to access in TF
    pred = [None]*numoutputs
    pred_node_names = [None]*numoutputs
    for i in range(numoutputs):
        pred_node_names[i] = prefix+'_'+str(i)
        pred[i] = tf.identity(net_model.outputs[i], name=pred_node_names[i])
    print('Output nodes names are: ', pred_node_names)
    
    # Write the graph in human readable
    aname = 'graph_def_for_reference.pb.ascii'
    graph_io.write_graph(sess.graph.as_graph_def(), outdir, aname, as_text=True)
    print('Saved the graph definition in ascii format at: ', os.path.join(outdir, aname))

    #freeze graph
    constant_graph = freeze_session(sess, output_names=pred_node_names)

    # Write the graph in binary .pb file
    graph_io.write_graph(constant_graph, outdir, name, as_text=False)
    print('Saved the constant graph (ready for inference) at: ', os.path.join(outdir, name))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model','-m', dest='model', required=True, help='REQUIRED: The HDF5 Keras model you wish to convert to .pb')
    parser.add_argument('--infer-model','-i', dest='infer_model', required=False, help='Inference model on which to copy weights to')
    parser.add_argument('--numout','-n', type=int, dest='num_out', required=True, help='REQUIRED: The number of outputs in the model.')
    parser.add_argument('--outdir','-o', dest='outdir', required=False, default='./', help='The directory to place the output files - default("./")')
    parser.add_argument('--prefix','-p', dest='prefix', required=False, default='k2tfout', help='The prefix for the output aliasing - default("k2tfout")')
    parser.add_argument('--name', dest='name', required=False, default='output_graph.pb', help='The name of the resulting output graph - default("output_graph.pb")')
    args = parser.parse_args()

    convertGraph( args.model, args.outdir, args.num_out, args.prefix, args.name, args.infer_model )

