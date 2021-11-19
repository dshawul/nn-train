from __future__ import print_function
import os
import argparse
import struct
import numpy as np
import matplotlib.pyplot as plt
from train import my_load_model, NNUE_KINDICES, NNUE_CHANNELS

#import tensorflow and set logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except:
    pass

SAVE_BIN = False
VERSION = 0

def freeze_model(model):
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    input_types = []
    for inp in model.inputs:
        input_types.append(inp.type_spec)
    full_model = full_model.get_concrete_function(input_types)

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    return frozen_func.graph

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

def plot(wm,name="",scale=8):
    """
    Plot weights
    """
    nr, nc = NNUE_KINDICES*8, 12*8
    wc = np.zeros(shape=(nr,nc,3), dtype=np.float32)
    for i in range(NNUE_KINDICES):
        for j in range(12):
            for k in range(64):
                r, g, b = 2.0, 2.0, 2.0
                for m in range(86):
                    r += wm[i*12*64+j*64+k,m]*scale
                for m in range(86):
                    g += wm[i*12*64+j*64+k,m+86]*scale
                for m in range(84):
                    b += wm[i*12*64+j*64+k,m+172]*scale
                r, g, b = r/4, g/4, b/4
                wc[(NNUE_KINDICES-1-i)*8 + (7-k//8), (j*8)+k%8, :] = [r,g,b]
                # print(r,g,b)

    im = plt.imshow(wc, interpolation='none')
    plt.colorbar(im,orientation="horizontal", fraction=0.046, pad=0.04)

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
            print(layer.__class__.__name__)
            for wi in w:
                print(wi.shape)
                wm = wi
                has_plot = False
                #add factorizer weights
                if wi.shape == (64*NNUE_CHANNELS, 256) and NNUE_KINDICES > 1:
                    win = wi.reshape(NNUE_CHANNELS, 64, 256)
                    wm = np.zeros(shape=(NNUE_KINDICES*12*64, 256), dtype=np.float32)
                    print(str(wm.shape) + " after resize")

                    plt.figure(figsize=(20, 10))

                    #no factorizer
                    for i in range(NNUE_KINDICES):
                        for j in range(12):
                            for k in range(64):
                                wm[i*12*64+j*64+k,:] =  wi[i*12*64+j*64+k,:]
                    plt.subplot(1,2,1)
                    plot(wm,'plain')

                    #k-psqt factorizer
                    for i in range(NNUE_KINDICES):
                        for j in range(12):
                            for k in range(64):
                                wm[i*12*64+j*64+k,:] =  \
                                        wi[i*12*64+j*64+k,:] + \
                                        wi[NNUE_KINDICES*12*64+j*64+k,:]
                    plt.subplot(1,2,2)
                    plot(wm,'kpsqt')
                    
                    has_plot = True

                if has_plot:
                    plt.savefig('weights.png')
                    # plt.show()

                #save weiights
                wf = wm.flatten()
                for x in wf:
                    data = struct.pack('f',x)
                    myF.write(data)

def convertGraph(modelPath, inferPath, outdir, name):

    from tensorflow.python.framework import graph_io

    tf.keras.backend.set_learning_phase(0)

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
        net_model = my_load_model(modelPath)
    else:
        net_model = my_load_model(modelPath)

    # My net format
    if SAVE_BIN:
        fpath = os.path.join(outdir, name + ".bin")
        save_weights(net_model, fpath)
        print('Raw weights: ', fpath)

    constant_graph = freeze_model(net_model)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=constant_graph,
                      logdir=outdir,
                      name=name+".pb",
                      as_text=False)
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

