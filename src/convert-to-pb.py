from __future__ import print_function
import os
import argparse
import struct
import numpy as np
import matplotlib.pyplot as plt
from train import (
    my_load_model,
    NNUE_KINDICES,
    NNUE_CHANNELS,
    NNUE_FEATURES,
    NNUE_FACTORIZER,
    NNUE_FACTORIZER_EXTRA,
)
from nnue import FT_WIDTH

# import tensorflow and set logging level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except:
    pass
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

SAVE_BIN = True
VERSION = 0
PLOT = True


def freeze_model(model):
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2,
    )

    # Convert Keras model to ConcreteFunction
    model._name = ""
    model_func = tf.function(lambda x: model(x))
    input_types = []
    for inp in model.inputs:
        input_types.append(inp.type_spec)
    conc_func = model_func.get_concrete_function(input_types)
    frozen_func = convert_variables_to_constants_v2(conc_func)

    # print layers
    gp = frozen_func.graph.as_graph_def()
    print("-" * 50)
    for i, n in enumerate(gp.node):
        print(str(i) + ". " + n.name)
    print("-" * 50)

    return gp


def copy_weights(m1, m2):
    """
    Copy weights from one model to another
    """
    for layer1, layer2 in zip(m1.layers, m2.layers):
        w1 = layer1.get_weights()
        if len(w1) > 0:
            w2 = layer2.get_weights()
            for k in range(len(w2)):
                w2[k] = w1[k].reshape(w2[k].shape)
            layer2.set_weights(w2)


def plot(wm, scale=8):
    """
    Plot weights
    """
    if not PLOT:
        return

    nr, nc = NNUE_KINDICES * 8, 12 * 8
    wc = np.zeros(shape=(nr, nc, 3), dtype=np.float32)
    for k in range(64):
        for i in range(NNUE_KINDICES):
            for j in range(12):
                r, g, b = 2.0, 2.0, 2.0
                r += np.sum(wm[k * NNUE_KINDICES * 12 + i * 12 + j, :86]) * scale
                g += np.sum(wm[k * NNUE_KINDICES * 12 + i * 12 + j, 86:172]) * scale
                b += np.sum(wm[k * NNUE_KINDICES * 12 + i * 12 + j, 172:]) * scale
                r, g, b = (
                    max(0, min(1, 1 - r / 4)),
                    max(0, min(1, 1 - g / 4)),
                    max(0, min(1, 1 - b / 4)),
                )
                wc[(NNUE_KINDICES - 1 - i) * 8 + (7 - k // 8), (j * 8) + k % 8, :] = [
                    r,
                    g,
                    b,
                ]

    im = plt.imshow(wc, interpolation="none")
    plt.colorbar(im, orientation="horizontal", fraction=0.046, pad=0.04)


def save_weights(m, name):
    """
    Save weights in raw format
    """
    myF = open(name, "wb")
    data = struct.pack("i", VERSION)
    myF.write(data)
    for layer in m.layers:
        w = layer.get_weights()
        if len(w) > 0:
            print(layer.__class__.__name__)
            for wi in w:
                print(wi.shape)
                wm = wi
                has_plot = False
                # add factorizer weights
                if wi.shape == (NNUE_FEATURES, FT_WIDTH) and NNUE_KINDICES > 1:
                    win = np.moveaxis(wi.reshape(64, NNUE_CHANNELS, FT_WIDTH), 1, 0)
                    wm = np.zeros(
                        shape=(NNUE_KINDICES * 12 * 64, FT_WIDTH), dtype=np.float32
                    )
                    wms = np.zeros(
                        shape=(NNUE_KINDICES * 12 * 64, FT_WIDTH), dtype=np.float32
                    )
                    print(str(wm.shape) + " after resize")

                    plt.figure(figsize=(40, 10))

                    n_plots = (
                        2
                        + (1 if NNUE_FACTORIZER else 0)
                        + (2 if NNUE_FACTORIZER_EXTRA else 0)
                    )

                    # no factorizer
                    for k in range(64):
                        for i in range(NNUE_KINDICES):
                            for j in range(12):
                                wm[k * NNUE_KINDICES * 12 + i * 12 + j, :] = wi[
                                    k * NNUE_CHANNELS + i * 12 + j, :
                                ]
                    ax = plt.subplot(1, n_plots, 1)
                    ax.title.set_text("plain")
                    plot(wm)

                    # k-psqt factorizer
                    if NNUE_FACTORIZER > 0:
                        wms = np.copy(wm)
                        for k in range(64):
                            for i in range(NNUE_KINDICES):
                                for j in range(12):
                                    wm[k * NNUE_KINDICES * 12 + i * 12 + j, :] += wi[
                                        k * NNUE_CHANNELS + NNUE_KINDICES * 12 + j, :
                                    ]
                        ax = plt.subplot(1, n_plots, 2)
                        ax.title.set_text("king factorizer")
                        plot(np.subtract(wm, wms))

                    # file,rank,and 4 rings factorizors
                    if NNUE_FACTORIZER_EXTRA > 0:
                        wms = np.copy(wm)
                        ch = (NNUE_KINDICES + 1) * 12
                        for k in range(64):
                            f = k % 8
                            r = k // 8
                            for i in range(NNUE_KINDICES):
                                for j in range(6):
                                    wm[k * NNUE_KINDICES * 12 + i * 12 + j, :] += (
                                        win[ch + 0, j * 8 + f, :]
                                        + win[ch + 1, j * 8 + r, :]
                                    )
                                    if r >= 1 and r < 7 and f >= 1 and f < 7:
                                        wm[
                                            k * NNUE_KINDICES * 12 + i * 12 + j, :
                                        ] += win[ch + 0, 7 * 8 + j, :]
                                    if r >= 2 and r < 6 and f >= 2 and f < 6:
                                        wm[
                                            k * NNUE_KINDICES * 12 + i * 12 + j, :
                                        ] += win[ch + 1, 6 * 8 + j, :]
                                    if r >= 3 and r < 5 and f >= 3 and f < 5:
                                        wm[
                                            k * NNUE_KINDICES * 12 + i * 12 + j, :
                                        ] += win[ch + 1, 7 * 8 + j, :]
                                    if (r + f) % 2 == 0:
                                        if j == 3:
                                            wm[
                                                k * NNUE_KINDICES * 12 + i * 12 + j, :
                                            ] += win[ch + 0, 6 * 8 + 6, :]
                                        if j == 4:
                                            wm[
                                                k * NNUE_KINDICES * 12 + i * 12 + j, :
                                            ] += win[ch + 0, 6 * 8 + 7, :]
                                        if j == 5:
                                            wm[
                                                k * NNUE_KINDICES * 12 + i * 12 + j, :
                                            ] += win[ch + 0, 7 * 8 + 6, :]
                                            if r >= 2 and r < 6 and f >= 2 and f < 6:
                                                wm[
                                                    k * NNUE_KINDICES * 12 + i * 12 + j,
                                                    :,
                                                ] += win[ch + 0, 7 * 8 + 7, :]
                                    if j == 3:
                                        if r == f:
                                            wm[
                                                k * NNUE_KINDICES * 12 + i * 12 + j, :
                                            ] += win[ch + 1, 6 * 8 + 6, :]
                                        if r + f == 7:
                                            wm[
                                                k * NNUE_KINDICES * 12 + i * 12 + j, :
                                            ] += win[ch + 1, 6 * 8 + 7, :]
                                        if r == f + 1 or r + 1 == f:
                                            wm[
                                                k * NNUE_KINDICES * 12 + i * 12 + j, :
                                            ] += win[ch + 1, 7 * 8 + 6, :]
                                        if r + f == 6 or r + f == 8:
                                            wm[
                                                k * NNUE_KINDICES * 12 + i * 12 + j, :
                                            ] += win[ch + 1, 7 * 8 + 7, :]
                        ax = plt.subplot(1, n_plots, 3)
                        ax.title.set_text("rfdc factorizers")
                        plot(np.subtract(wm, wms))

                        # material
                        wms = np.copy(wm)
                        for k in range(64):
                            for i in range(NNUE_KINDICES):
                                for j in range(6):
                                    wm[k * NNUE_KINDICES * 12 + i * 12 + j, :] += win[
                                        ch + 0, 6 * 8 + j, :
                                    ]
                        ax = plt.subplot(1, n_plots, 4)
                        ax.title.set_text("material factorizer")
                        plot(np.subtract(wm, wms), scale=4)

                        # total
                        ax = plt.subplot(1, n_plots, 5)
                        ax.title.set_text("total")
                        plot(wm, scale=4)

                    has_plot = True

                if has_plot:
                    plt.savefig("weights.png")

                # save weiights
                wf = wm.flatten()
                for x in wf:
                    data = struct.pack("f", x)
                    myF.write(data)


def convertGraph(modelPath, inferPath, outdir, name):

    from tensorflow.python.framework import graph_io

    tf.keras.backend.set_learning_phase(0)

    # defaults
    if outdir == None:
        outdir = os.path.dirname(modelPath)
    if name == None:
        name = os.path.basename(modelPath)

    # create dir
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    if inferPath != None:
        # copy weights
        opt_model = my_load_model(modelPath)
        net_model = my_load_model(inferPath)
        copy_weights(opt_model, net_model)
    else:
        net_model = my_load_model(modelPath)

    # My net format
    if SAVE_BIN:
        fpath = os.path.join(outdir, name + ".bin")
        save_weights(net_model, fpath)
        print("Raw weights: ", fpath)

    constant_graph = freeze_model(net_model)

    # Save frozen graph from frozen ConcreteFunction
    tf.io.write_graph(
        graph_or_graph_def=constant_graph,
        logdir=outdir,
        name=name + ".pb",
        as_text=False,
    )
    path_pb = os.path.join(outdir, name + ".pb")
    print("Frozen protobuf graph: ", path_pb)

    # output names
    if "main_input" in net_model.input_names:
        output_nodes = ["policy/Reshape", "value/BiasAdd"]
        input_nodes = ["main_input"]
    else:
        output_nodes = ["valuea/Sigmoid"]
        input_nodes = ["player_input", "opponent_input"]

    # Save onnx model
    try:
        from tf2onnx import tf_loader
        from tf2onnx.tfonnx import process_tf_graph
        from tf2onnx.optimizer import optimize_graph
        from tf2onnx import utils, constants
        from tf2onnx.handler import tf_op

        extra_opset = [utils.make_opsetid(constants.CONTRIB_OPS_DOMAIN, 1)]
        tensors_to_rename = {f"{e}:0": e for e in input_nodes}
        tensors_to_rename.update({f"{e}:0": e for e in output_nodes})
        with tf.Graph().as_default() as tf_graph:
            tf.import_graph_def(constant_graph, name="")
        with tf_loader.tf_session(graph=tf_graph):
            g = process_tf_graph(
                tf_graph,
                output_names=[f"{e}:0" for e in output_nodes],
                inputs_as_nchw=[f"{e}:0" for e in input_nodes],
                tensors_to_rename=tensors_to_rename,
                extra_opset=extra_opset,
            )
        onnx_graph = optimize_graph(g)
        model_proto = onnx_graph.make_model("converted")
        path_onnx = os.path.join(outdir, name + ".onnx")
        utils.save_protobuf(path_onnx, model_proto)
        print("Fronzen ONNX graph: ", path_onnx)
    except Exception as e:
        print(e)


    # Save uff model
    try:
        import uff

        path_uff = os.path.join(outdir, name + ".uff")
        uff.from_tensorflow(
            constant_graph,
            output_nodes=output_nodes,
            output_filename=path_uff,
            write_preprocessed=False,
            text=False,
            debug_mode=False,
            return_graph_info=False,
        )
        print("Frozen UFF graph: ", path_uff)
    except Exception as e:
        print(e)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        dest="model",
        required=True,
        help="REQUIRED: The HDF5 Keras model you wish to convert to .pb",
    )
    parser.add_argument(
        "--infer-model",
        "-i",
        dest="infer_model",
        required=False,
        help="Inference model on which to copy weights to",
    )
    parser.add_argument(
        "--outdir",
        "-o",
        dest="outdir",
        required=False,
        default=None,
        help="The directory to place the output files",
    )
    parser.add_argument(
        "--name",
        "-n",
        dest="name",
        required=False,
        default=None,
        help="The name of the resulting output graph",
    )
    args = parser.parse_args()

    convertGraph(args.model, args.infer_model, args.outdir, args.name)
