from __future__ import print_function
import sys
import time
import chess
import resnet
import argparse
import gzip
import numpy as np
import multiprocessing as mp
from joblib import Parallel, delayed

#import tensorflow and set logging level
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except:
    pass

#global params
AUX_INP = True
CHANNELS = 32
BOARDX = 8
BOARDY = 8
POLICY_CHANNELS = 16
NBATCH = 512
BATCH_SIZE = 512
EPD_CHUNK_SIZE = BATCH_SIZE * NBATCH
PIECE_MAP = "KQRBNPkqrbnp"
RANK_U = BOARDY - 1
FILE_U = BOARDX - 1
FRAC_PI = 1
FRAC_Z  = 1
HEAD_TYPE = 0

def fill_piece(iplanes, ix, bb, b, flip_file):

    if AUX_INP:
        abb = 0
        squares = chess.SquareSet(bb)
        for sq in squares:
            abb = abb | b.attacks_mask(sq)
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            if b.turn == chess.BLACK: r = RANK_U -r
            if flip_file: f = FILE_U - f
            iplanes[r,  f,  ix + 12] = 1.0

        squares = chess.SquareSet(abb)
        for sq in squares:
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            if b.turn == chess.BLACK: r = RANK_U -r
            if flip_file: f = FILE_U - f
            iplanes[r,  f,  ix] = 1.0
    else:
        squares = chess.SquareSet(bb)
        for sq in squares:
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            if b.turn == chess.BLACK: r = RANK_U -r
            if flip_file: f = FILE_U - f
            iplanes[r,  f,  ix] = 1.0

def fill_planes(iplanes, b):
    pl = b.turn
    npl = not b.turn

    #flip horizontal
    flip_file = (chess.square_file(b.king(pl)) < 4)

    #white piece attacks
    bb = b.kings   & b.occupied_co[pl]
    fill_piece(iplanes,0,bb,b,flip_file)
    bb = b.queens  & b.occupied_co[pl]
    fill_piece(iplanes,1,bb,b,flip_file)
    bb = b.rooks   & b.occupied_co[pl]
    fill_piece(iplanes,2,bb,b,flip_file)
    bb = b.bishops & b.occupied_co[pl]
    fill_piece(iplanes,3,bb,b,flip_file)
    bb = b.knights & b.occupied_co[pl]
    fill_piece(iplanes,4,bb,b,flip_file)
    bb = b.pawns   & b.occupied_co[pl]
    fill_piece(iplanes,5,bb,b,flip_file)

    #black piece attacks
    bb = b.kings   & b.occupied_co[npl]
    fill_piece(iplanes,6,bb,b,flip_file)
    bb = b.queens  & b.occupied_co[npl]
    fill_piece(iplanes,7,bb,b,flip_file)
    bb = b.rooks   & b.occupied_co[npl]
    fill_piece(iplanes,8,bb,b,flip_file)
    bb = b.bishops & b.occupied_co[npl]
    fill_piece(iplanes,9,bb,b,flip_file)
    bb = b.knights & b.occupied_co[npl]
    fill_piece(iplanes,10,bb,b,flip_file)
    bb = b.pawns   & b.occupied_co[npl]
    fill_piece(iplanes,11,bb,b,flip_file)

    #enpassant, casling, fifty and on-board mask
    if b.ep_square:
        f = chess.square_file(b.ep_square)
        r = chess.square_rank(b.ep_square)
        if b.turn == chess.BLACK: r = RANK_U -r
        if flip_file: f = FILE_U - f
        iplanes[r, f, CHANNELS - 8] = 1.0

    if b.has_queenside_castling_rights(pl):
        iplanes[:, :, CHANNELS - (6 if flip_file else 7)] = 1.0
    if b.has_kingside_castling_rights(pl):
        iplanes[:, :, CHANNELS - (7 if flip_file else 6)] = 1.0
    if b.has_queenside_castling_rights(npl):
        iplanes[:, :, CHANNELS - (4 if flip_file else 5)] = 1.0
    if b.has_kingside_castling_rights(npl):
        iplanes[:, :, CHANNELS - (5 if flip_file else 4)] = 1.0

    iplanes[:, :, CHANNELS - 3] = b.fullmove_number / 200.0
    iplanes[:, :, CHANNELS - 2] = b.halfmove_clock / 100.0
    iplanes[:, :, CHANNELS - 1] = 1.0

def fill_planes_fen(iplanes, fen, player):

    #board
    kf = 4
    cnt = 0
    for r in range(RANK_U, -1, -1):
        for f in range(0, FILE_U + 1, 1):
            c = fen[cnt]
            idx = PIECE_MAP.find(c)
            if idx != -1:
                if player == 0:
                    iplanes[r,  f,  idx] = 1.0
                else:
                    iplanes[RANK_U - r,  f,  (idx^1)] = 1.0
                if c == 'K':
                    kf = f
                cnt = cnt + 1
            elif c.isdigit():
                b = int(c)
                cnt = cnt + 1
                c = fen[cnt]
                if c.isdigit():
                    cnt = cnt + 1
                f = f + b - 1
            else:
                break
        cnt = cnt + 1

    #flip file
    flip_file = (kf < 4)

    if flip_file:
        for f in range(0, FILE_U + 1, 1):
            temp = iplanes[:, FILE_U - f, :]
            iplanes[:, FILE_U - f, :] = iplanes[:, f, :]
            iplanes[:, f, :] = temp

    #holdings
    if fen[cnt - 1] == '[':

        N_PIECES = len(PIECE_MAP)

        if fen[cnt] == '-':
            cnt = cnt + 1
        else:
            holdings = np.zeros(N_PIECES)

            while True:
                idx = PIECE_MAP.find(fen[cnt])
                if idx == -1:
                    break
                holdings[idx] = holdings[idx] + 1
                cnt = cnt + 1

            for idx in range(N_PIECES):
                if holdings[idx] > 0:
                    if player == 0:
                        iplanes[:,  :,    idx   + N_PIECES] = holdings[idx] / 50.0
                    else:
                        iplanes[:,  :,  (idx^1) + N_PIECES] = holdings[idx] / 50.0
        cnt = cnt + 2

    #enpassant, castling, fifty and on-board mask
    epstr = words[2]
    if epstr[0] != '-':
        f = epstr[0] - 'a'
        r = epstr[1] - '1'
        if player == 1: r = RANK_U -r
        if flip_file: f = FILE_U - f
        iplanes[r, f, CHANNELS - 8] = 1.0

    castr = words[1]
    if castr.find('Q' if player == 0 else 'q') != -1:
        iplanes[:, :, CHANNELS - (6 if flip_file else 7)] = 1.0
    if castr.find('K' if player == 0 else 'k') != -1:
        iplanes[:, :, CHANNELS - (7 if flip_file else 6)] = 1.0
    if castr.find('q' if player == 0 else 'Q') != -1:
        iplanes[:, :, CHANNELS - (4 if flip_file else 5)] = 1.0
    if castr.find('k' if player == 0 else 'K') != -1:
        iplanes[:, :, CHANNELS - (5 if flip_file else 4)] = 1.0

    iplanes[:, :, CHANNELS - 3] = float(words[4]) / 200.0
    iplanes[:, :, CHANNELS - 2] = float(words[3]) / 100.0
    iplanes[:, :, CHANNELS - 1] = 1.0


def fill_examples(examples):
    if AUX_INP:
        bb = chess.Board()

    N = len(examples)
    iplanes = np.zeros(shape=(N,BOARDY,BOARDX,CHANNELS),dtype=np.float32)
    oresult = np.zeros(shape=(N,),dtype=np.int)
    ovalue = np.zeros(shape=(N,3),dtype=np.float32)
    if HEAD_TYPE == 0:
        opolicy = np.zeros(shape=(N,POLICY_CHANNELS*BOARDX*BOARDY),dtype=np.float32)
        ret = [iplanes, oresult, ovalue, opolicy]
    elif HEAD_TYPE == 1:
        oscore = np.zeros(shape=(N,POLICY_CHANNELS*BOARDX*BOARDY),dtype=np.float32)
        ret = [iplanes, oresult, ovalue, oscore]
    else:
        opolicy = np.zeros(shape=(N,POLICY_CHANNELS*BOARDX*BOARDY),dtype=np.float32)
        oscore = np.zeros(shape=(N,POLICY_CHANNELS*BOARDX*BOARDY),dtype=np.float32)
        ret = [iplanes, oresult, ovalue, opolicy, oscore]

    for id,line in enumerate(examples):

        words = line.strip().split()

        epd = ''

        for i in range(0, 6):
            epd = epd + words[i] + ' '

        #player
        if words[1] == 'b':
            player = 1
        else:
            player = 0

        # parse result
        svalue = words[6]
        if svalue == '1-0':
            result = 0
        elif svalue == '0-1':
            result = 2
        else:
            result = 1

        # value
        value = float(words[7])

        # nmoves
        nmoves = int(words[8])

        #set board
        if AUX_INP:
            bb.set_fen(epd)

        offset = 9

        #flip board
        if player == 1:
            result = 2 - result
            value = 1 - value

        #result
        oresult[id] = result

        #value
        if FRAC_Z == 1:
            ovalue[id,:] = 0.0
            ovalue[id,result] = 1.0
        else:
            ovalue[id,1] = 0.7 * min(value, 1 - value)
            ovalue[id,0] = value - ovalue[id,1] / 2.0
            ovalue[id,2] = 1 - ovalue[id,0] - ovalue[id,1]

            if FRAC_Z > 0:
                ovalue[id,:] *= (1 - FRAC_Z)
                ovalue[id,result] += FRAC_Z

        #policy
        V = (1.0 - result / 2.0)
        V=max(V,1e-5)

        if HEAD_TYPE == 0:
            for i in range(offset, offset+nmoves*2, 2):
                opolicy[id, int(words[i])] = float(words[i+1])
            offset += nmoves*2
        elif HEAD_TYPE == 1:
            for i in range(offset, offset+nmoves*2, 2):
                oscore[id, int(words[i])] = float(words[i+1])
            offset += nmoves*2
        else:
            for i in range(offset, offset+nmoves*3, 3):
                opolicy[id, int(words[i])] = float(words[i+1])
                oscore[id, int(words[i])] = float(words[i+2])
            offset += nmoves*3

        if (FRAC_PI < 1) and (offset < len(words)):
            bestm = int(words[offset])
            opolicy[id, :] *= FRAC_PI
            if HEAD_TYPE == 0:
                opolicy[id, bestm] += (1 - FRAC_PI)
            elif HEAD_TYPE == 1:
                oscore[id, bestm] += (1 - FRAC_PI) * V
            else:
                opolicy[id, bestm] += (1 - FRAC_PI)
                oscore[id, bestm] += (1 - FRAC_PI) * V

        #input planes
        if AUX_INP:
            fill_planes(iplanes[id,:,:,:],bb)
        else:
            fill_planes_fen(iplanes[id,:,:,:],epd,words,player)

    return ret

def build_model(cid):
    INPUT_SHAPE=(None, None, CHANNELS)
    if cid == 0:
        return resnet.build_net(INPUT_SHAPE,  2,  32, POLICY_CHANNELS, HEAD_TYPE)
    elif cid == 1:
        return resnet.build_net(INPUT_SHAPE,  6,  64, POLICY_CHANNELS, HEAD_TYPE)
    elif cid == 2:
        return resnet.build_net(INPUT_SHAPE, 12, 128, POLICY_CHANNELS, HEAD_TYPE)
    elif cid == 3:
        return resnet.build_net(INPUT_SHAPE, 20, 256, POLICY_CHANNELS, HEAD_TYPE)
    elif cid == 4:
        return resnet.build_net(INPUT_SHAPE, 40, 256, POLICY_CHANNELS, HEAD_TYPE)
    else:
        print("Unsupported network id (Use 0 to 4).")
        sys.exit()

class NNet():
    def __init__(self,args):
        self.mirrored_strategy = tf.distribute.MirroredStrategy()

    def new_model(self,idx,args):
        if args.gpus > 1:
            with self.mirrored_strategy.scope():
                return build_model(idx)
        else:
            return build_model(idx)

    def load_model(self,fname,compile,args):
        if args.gpus > 1:
            with self.mirrored_strategy.scope():
                return tf.keras.models.load_model(fname, compile=compile)
        else:
            return tf.keras.models.load_model(fname, compile=compile)

    def compile_model(self,mdx,args):
        if args.opt == 0:
            opt = tf.keras.optimizers.SGD(lr=args.lr,momentum=0.9,nesterov=True)
        else:
            opt = tf.keras.optimizers.Adam(lr=args.lr)

        if args.mixed:
            opt = tf.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(opt)

        # losses and accuracy
        def loss(y_true, y_pred):
            is_legal = tf.greater(y_true, 0)
            y_pred = tf.where(is_legal, y_pred, y_true)
            return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

        def accuracy(y_true,y_pred):
            is_legal = tf.greater(y_true, 0)
            y_pred = tf.where(is_legal, y_pred, y_true)
            return tf.keras.metrics.categorical_accuracy(y_true, y_pred)

        def sloss(y_true, y_pred):
            is_legal = tf.greater(y_true, 0)
            y_pred = tf.where(is_legal, y_pred, y_true)

            su = tf.reduce_sum(tf.cast(is_legal, tf.float32))
            sz = tf.cast(tf.size(is_legal), tf.float32)
            return  (sz / su) * tf.keras.losses.mean_squared_error(y_true, y_pred)

        if HEAD_TYPE == 0:
            losses = {"value":'categorical_crossentropy', "policya":loss}
            metrics = {"value":'accuracy', "policya":accuracy}
            loss_weights = [args.val_w, args.pol_w]

        elif HEAD_TYPE == 1:
            losses  = {"value":'categorical_crossentropy', "scorea":sloss}
            metrics = {"value":'accuracy'}
            loss_weights = [args.val_w, args.score_w]

        else:
            losses  = {"value":'categorical_crossentropy', "policya":loss, "scorea":sloss}
            metrics = {"value":'accuracy', "policya":accuracy}
            loss_weights = [args.val_w, args.pol_w, args.score_w]

        # compile model
        if args.gpus > 1:
            with self.mirrored_strategy.scope():
                mdx.compile(loss=losses,loss_weights=loss_weights,
                      optimizer=opt,metrics=metrics)
        else:
            mdx.compile(loss=losses,loss_weights=loss_weights,
                  optimizer=opt,metrics=metrics)

    def train(self,examples,local_steps,args):
        print("Generating input planes using", args.cores, "cores")
        start_t = time.time()

        N = len(examples)

        #multiprocess
        nlen = N / args.cores
        slices = [ slice((id*nlen) , (min(N,(id+1)*nlen))) for id in range(args.cores) ]
        res = Parallel(n_jobs=args.cores)( delayed(fill_examples) (examples[sl]) for sl in slices )

        #accumulate
        ipln = np.zeros(shape=(N,BOARDY,BOARDX,CHANNELS),dtype=np.float32)
        ores = np.zeros(shape=(N,),dtype=np.int)
        oval = np.zeros(shape=(N,3),dtype=np.float32)
        if HEAD_TYPE == 0:
            opol = np.zeros(shape=(N,POLICY_CHANNELS*BOARDX*BOARDY),dtype=np.float32)
            y = [oval, opol]
        elif HEAD_TYPE == 1:
            osco = np.zeros(shape=(N,POLICY_CHANNELS*BOARDX*BOARDY),dtype=np.float32)
            y = [oval, osco]
        else:
            opol = np.zeros(shape=(N,POLICY_CHANNELS*BOARDX*BOARDY),dtype=np.float32)
            osco = np.zeros(shape=(N,POLICY_CHANNELS*BOARDX*BOARDY),dtype=np.float32)
            y = [oval, opol, osco]

        for i in range(args.cores):
            ipln[slices[i],:,:,:] = res[i][0]
            ores[slices[i]] = res[i][1]
            oval[slices[i],:] = res[i][2]
            if HEAD_TYPE == 0:
                opol[slices[i],:] = res[i][3]
            elif HEAD_TYPE == 1:
                osco[slices[i],:] = res[i][3]
            else:
                opol[slices[i],:] = res[i][3]
                osco[slices[i],:] = res[i][4]

        end_t = time.time()
        print("Time", int(end_t - start_t), "sec")

        start_t = end_t
        vweights = None
        pweights = None
        if args.pol_grad > 0:
            vweights = np.ones(ores.size)
            pweights = (1.0 - ores / 2.0) - (oval[:,0] + oval[:,1] / 2.0) #  Z-Q

        for i in range(len(self.model)):
            print("Fitting model",i)

            tensorboard_callback = self.cbacks[i]
            initial_epoch = args.global_steps + local_steps
            epochs = initial_epoch + args.epochs

            if args.pol_grad > 0:
                self.model[i].fit(x = [ipln], y = y,
                      batch_size=BATCH_SIZE,
                      sample_weight=[vweights, pweights],
                      validation_split=args.vald_split,
                      initial_epoch=initial_epoch,
                      epochs=epochs,
                      callbacks=[tensorboard_callback])
            else:
                self.model[i].fit(x = [ipln], y = y,
                      batch_size=BATCH_SIZE,
                      validation_split=args.vald_split,
                      initial_epoch=initial_epoch,
                      epochs=epochs,
                      callbacks=[tensorboard_callback])

        end_t = time.time()
        print("Training time", int(end_t - start_t), "sec")

    def save_checkpoint(self, chunk, args, iopt=False):
        filepath = os.path.join(args.dir, "ID-" + str(chunk))
        if not os.path.exists(args.dir):
            os.mkdir(args.dir)

        #save each model
        for i,n in enumerate(args.nets):
            fname = filepath  + "-model-" + str(n)
            self.model[i].save(fname, include_optimizer=iopt, save_format='h5')

    def load_checkpoint(self, chunk, args):
        filepath = os.path.join(args.dir, "ID-" + str(chunk))

        #create training model
        self.model = []
        self.cbacks = []
        for n in args.nets:
            fname = filepath  + "-model-" + str(n)
            if not os.path.exists(fname):
                mdx = self.new_model(n,args)
                self.compile_model(mdx, args)
            else:
                comp = (chunk % args.rsavo == 0)
                mdx = self.load_model(fname,comp,args)
                if not mdx.optimizer:
                    print("====== ", fname, " : starting from fresh optimizer state ======")
                    self.compile_model(mdx, args)
            self.model.append( mdx )

            log_dir = "logs/fit/model-" + str(n)
            tensorboard_callback = tf.keras.callbacks.TensorBoard( \
                log_dir=log_dir,update_freq='epoch')
            self.cbacks.append ( tensorboard_callback )

    def save_infer_graph(self, args):
        filepath = os.path.join(args.dir, "infer-")
        if not os.path.exists(args.dir):
            os.mkdir(args.dir)

        tf.keras.backend.set_learning_phase(0)

        #create inference model
        for n in args.nets:
            fname = filepath + str(n)
            new_model = self.new_model(n,args)
            new_model.save(fname, include_optimizer=False, save_format='h5')

        tf.keras.backend.clear_session()
        tf.keras.backend.set_learning_phase(1)

def train_epd(myNet,args,myEpd,start=1):

    with (open(myEpd) if not args.gzip else gzip.open(myEpd)) as file:
        count = 0

        examples = []
        start_t = time.time()
        print("Collecting data")

        while True:

            #read fen
            line = file.readline()
            count = count + 1
            if count < start:
                continue

            # add to examples
            if line:
                examples.append(line)
            else:
                count = count - 1

            #train network
            if (not line) or (count % EPD_CHUNK_SIZE == 0):
                chunk = (count + EPD_CHUNK_SIZE - 1) / EPD_CHUNK_SIZE
                end_t = time.time()

                #make sure size is divisible by BATCH_SIZE
                if (not line) and (count % BATCH_SIZE != 0): 
                   count = (count//BATCH_SIZE)*BATCH_SIZE
                   if count > 0:
                        del examples[count:]
                   else:
                        break

                if len(examples) > 0:
                    print("Time", int(end_t - start_t), "sec")
                    print("Training on chunk ", chunk , " ending at position ", count, " with lr ", args.lr)
                    myNet.train(examples,(chunk-1)*NBATCH,args)

                    start_t = time.time()
                    if chunk % args.rsavo == 0:
                        myNet.save_checkpoint(chunk, args, True)
                    elif chunk % args.rsav == 0:
                        myNet.save_checkpoint(chunk, args, False)
                    end_t = time.time()
                    print("Saving time", int(end_t - start_t), "sec")

                    start_t = time.time()
                    examples = []
                    print("Collecting data")

            #break out
            if not line:
                break

def main(argv):
    global AUX_INP, EPD_CHUNK_SIZE, CHANNELS, BOARDX, BOARDY, FRAC_Z, FRAC_PI
    global POLICY_CHANNELS,  NBATCH, BATCH_SIZE, PIECE_MAP, RANK_U, FILE_U, HEAD_TYPE

    parser = argparse.ArgumentParser()
    parser.add_argument('--epd','-e', dest='epd', required=False, help='Path to labeled EPD file for training')
    parser.add_argument('--dir', dest='dir', required=False, default="nets", help='Path to network files')
    parser.add_argument('--id','-i', dest='id', required=False, type=int, default=0, help='ID of neural networks to load.')
    parser.add_argument('--global-steps', dest='global_steps', required=False, type=int, default=0, help='Global number of steps trained so far.')
    parser.add_argument('--batch-size','-b',dest='batch_size', required=False, type=int, default=BATCH_SIZE, help='Training batch size.')
    parser.add_argument('--nbatch',dest='nbatch', required=False, type=int, default=NBATCH, help='Number of batches to process at one time.')
    parser.add_argument('--epochs',dest='epochs', required=False, type=int, default=1, help='Training epochs.')
    parser.add_argument('--learning-rate','-l',dest='lr', required=False, type=float, default=0.01, help='Training learning rate.')
    parser.add_argument('--validation-split',dest='vald_split', required=False, type=float, default=0.125, help='Fraction of sample to use for validation.')
    parser.add_argument('--cores',dest='cores', required=False, type=int, default=mp.cpu_count(), help='Number of cores to use.')
    parser.add_argument('--gpus',dest='gpus', required=False, type=int, default=0, help='Number of gpus to use.')
    parser.add_argument('--gzip','-z',dest='gzip', required=False, action='store_true',help='Process zipped file.')
    parser.add_argument('--nets',dest='nets', nargs='+', required=False, type=int, default=[0,1,2], \
                        help='Nets to train from 0=2x32,6x64,12x128,20x256,4=40x256.')
    parser.add_argument('--rsav',dest='rsav', required=False, type=int, default=1, help='Save graph every RSAV chunks.')
    parser.add_argument('--rsavo',dest='rsavo', required=False, type=int, default=20, help='Save optimization state every RSAVO chunks.')
    parser.add_argument('--rand',dest='rand', required=False, action='store_true', help='Generate random network.')
    parser.add_argument('--opt',dest='opt', required=False, type=int, default=0, help='Optimizer 0=SGD 1=Adam.')
    parser.add_argument('--policy-channels',dest='pol_channels', required=False, type=int, default=POLICY_CHANNELS, help='Number of policy channels')
    parser.add_argument('--policy-weight',dest='pol_w', required=False, type=float, default=1.0, help='Policy loss weight.')
    parser.add_argument('--value-weight',dest='val_w', required=False, type=float, default=1.0, help='Value loss weight.')
    parser.add_argument('--score-weight',dest='score_w', required=False, type=float, default=1.0, help='Score loss weight.')
    parser.add_argument('--policy-gradient',dest='pol_grad', required=False, type=int, default=0, help='0=standard 1=multiply policy by score.')
    parser.add_argument('--no-auxinp','-u',dest='noauxinp', required=False, action='store_false', help='Don\'t use auxillary input')
    parser.add_argument('--channels','-c', dest='channels', required=False, type=int, default=CHANNELS, help='number of input channels of network.')
    parser.add_argument('--boardx','-x', dest='boardx', required=False, type=int, default=BOARDX, help='board x-dimension.')
    parser.add_argument('--boardy','-y', dest='boardy', required=False, type=int, default=BOARDY, help='board y-dimension.')
    parser.add_argument('--frac-z',dest='frac_z', required=False, type=float, default=FRAC_Z, help='Fraction of ouctome(Z) relative to MCTS value(Q).')
    parser.add_argument('--frac-pi',dest='frac_pi', required=False, type=float, default=FRAC_PI, \
        help='Fraction of MCTS policy (PI) relative to one-hot policy(P).')
    parser.add_argument('--piece-map',dest='pcmap', required=False, default=PIECE_MAP,help='Map pieces to planes')
    parser.add_argument('--mixed', dest='mixed', required=False, action='store_true', help='Use mixed precision training')
    parser.add_argument('--head-type',dest='head_type', required=False, type=int, default=HEAD_TYPE, \
        help='Heads of neural network, 0=value/policy, 1=value/score, 2=all three.')

    args = parser.parse_args()

    tf.keras.backend.set_learning_phase(1)

    #memory growth of gpus
    if args.gpus:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # init net
    myNet = NNet(args)

    NBATCH = args.nbatch
    BATCH_SIZE = args.batch_size
    EPD_CHUNK_SIZE = BATCH_SIZE * NBATCH
    CHANNELS = args.channels
    BOARDX = args.boardx
    BOARDY = args.boardy
    RANK_U = BOARDY - 1
    FILE_U = BOARDX - 1
    POLICY_CHANNELS = args.pol_channels
    AUX_INP = args.noauxinp
    FRAC_Z = args.frac_z
    FRAC_PI = args.frac_pi
    PIECE_MAP = args.pcmap
    HEAD_TYPE = args.head_type

    chunk = args.id

    #save inference graphs
    myNet.save_infer_graph(args)

    #initialize mixed precision training
    if args.mixed:
        config = tf.compat.v1.ConfigProto()
        config.graph_options.rewrite_options.auto_mixed_precision = True
        sess = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(sess)

    #load networks
    print("Loadng networks")
    start_t = time.time()
    myNet.load_checkpoint(chunk, args)
    end_t = time.time()
    print("Time", int(end_t - start_t), "sec")

    if args.rand:
        myNet.save_checkpoint(chunk, args)
    else:
        start = chunk * EPD_CHUNK_SIZE + 1
        train_epd(myNet, args, args.epd, start)


if __name__ == "__main__":
    main(sys.argv[1:])
