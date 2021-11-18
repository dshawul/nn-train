from __future__ import print_function
import sys
import time
import chess
import resnet
import nnue
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
MAX_QUEUE = 1

#NNUE
NNUE_KIDX = 4
NNUE_FACTORIZER_EXTRA = 2
NNUE_KINDICES = (1 << NNUE_KIDX)
NNUE_FACTORIZER = (12 if NNUE_KIDX > 0 else 0) + NNUE_FACTORIZER_EXTRA
NNUE_CHANNELS = NNUE_KINDICES*12 + NNUE_FACTORIZER  # NNUE_CHANNELS used during training
                                                    # CHANNELS=12 is used during data processing

NNUE_KINDEX_TAB = [
   [
    0,  0,  0,  0,
    0,  0,  0,  0,
    0,  0,  0,  0,
    0,  0,  0,  0,
    0,  0,  0,  0,
    0,  0,  0,  0,
    0,  0,  0,  0,
    0,  0,  0,  0
   ],
   [
    0,  0,  0,  0,
    1,  1,  1,  1,
    1,  1,  1,  1,
    1,  1,  1,  1,
    1,  1,  1,  1,
    1,  1,  1,  1,
    1,  1,  1,  1,
    1,  1,  1,  1
   ],
   [
    0,  0,  1,  1,
    2,  2,  2,  2,
    3,  3,  3,  3,
    3,  3,  3,  3,
    3,  3,  3,  3,
    3,  3,  3,  3,
    3,  3,  3,  3,
    3,  3,  3,  3
   ],
   [
    0,  1,  2,  3,
    4,  4,  5,  5,
    6,  6,  6,  6,
    7,  7,  7,  7,
    7,  7,  7,  7,
    7,  7,  7,  7,
    7,  7,  7,  7,
    7,  7,  7,  7
   ],
   [
    0,  1,  2,  3,
    4,  5,  6,  7,
    8,  8,  9,  9,
   10, 10, 11, 11,
   12, 12, 13, 13,
   12, 12, 13, 13,
   14, 14, 15, 15,
   14, 14, 15, 15
   ],
   [
    0,  1,  2,  3,
    4,  5,  6,  7,
    8,  9, 10, 11,
   12, 13, 14, 15,
   16, 17, 18, 19,
   20, 21, 22, 23,
   24, 25, 26, 27,
   28, 29, 30, 31
   ]
]

def fill_piece(iplanes, ix, bb, b, flip_rank, flip_file):
    """ Compute piece placement and attack plane for a given piece type """

    if AUX_INP:
        abb = 0
        squares = chess.SquareSet(bb)
        for sq in squares:
            abb = abb | b.attacks_mask(sq)
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            if flip_rank: r = RANK_U - r
            if flip_file: f = FILE_U - f
            iplanes[ix + 12, r,  f] = 1

        squares = chess.SquareSet(abb)
        for sq in squares:
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            if flip_rank: r = RANK_U - r
            if flip_file: f = FILE_U - f
            iplanes[ix, r,  f] = 1
    else:
        squares = chess.SquareSet(bb)
        for sq in squares:
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            if flip_rank: r = RANK_U - r
            if flip_file: f = FILE_U - f
            iplanes[ix, r,  f] = 1

            if HEAD_TYPE == 3 and NNUE_FACTORIZER_EXTRA != 0 and ix < 6:
                iplanes[CHANNELS + 0, ix, f] += 1                 #rows
                iplanes[CHANNELS + 1, ix, r] += 1                 #columns
                iplanes[CHANNELS + 0, 6, ix] += 1                 #ring-1 (material)
                if r >= 1 and r < 7 and f >= 1 and f < 7:
                    iplanes[CHANNELS + 0, 7, ix] += 1             #ring-2
                    if r >= 2 and r < 6 and f >= 2 and f < 6:
                        iplanes[CHANNELS + 1, 6, ix] += 1         #ring-3
                        if r >= 3 and r < 5 and f >= 3 and f < 5:
                            iplanes[CHANNELS + 1, 7, ix] += 1     #ring-4 (center)
                if ((r + f) & 1) == 0:
                    if ix == 3:
                        iplanes[CHANNELS + 0, 6, 6] += 1          #bishops on dark-square
                    if ix == 4:
                        iplanes[CHANNELS + 0, 6, 7] += 1          #knights on dark-square
                    if ix == 5:
                        iplanes[CHANNELS + 0, 7, 6] += 1          #pawns on dark-square
                        if r >= 2 and r < 6 and f >= 2 and f < 6:
                            iplanes[CHANNELS + 0, 7, 7] += 1      #center-pawns on dark-square
                if ix == 3:
                    if r == f:
                        iplanes[CHANNELS + 1, 6, 6] += 1          #bishop on a1-h8 diagonal
                    if r + f == 7:
                        iplanes[CHANNELS + 1, 6, 7] += 1          #bishop on h1-a8 diagonal
                    if r == f + 1 or r + 1 == f:
                        iplanes[CHANNELS + 1, 7, 6] += 1          #bishop on two diagonals closest to a1-h8
                    if r + f == 6 or r + f == 8:
                        iplanes[CHANNELS + 1, 7, 7] += 1          #bishop on two diagonals closest to h1-a8


def fill_planes_(iplanes, b, side, flip_rank, flip_file):
    """ Compute piece and attack planes for all pieces"""

    pl = side
    npl = not side

    #white piece attacks
    bb = b.kings   & b.occupied_co[pl]
    fill_piece(iplanes,0,bb,b,flip_rank,flip_file)
    bb = b.queens  & b.occupied_co[pl]
    fill_piece(iplanes,1,bb,b,flip_rank,flip_file)
    bb = b.rooks   & b.occupied_co[pl]
    fill_piece(iplanes,2,bb,b,flip_rank,flip_file)
    bb = b.bishops & b.occupied_co[pl]
    fill_piece(iplanes,3,bb,b,flip_rank,flip_file)
    bb = b.knights & b.occupied_co[pl]
    fill_piece(iplanes,4,bb,b,flip_rank,flip_file)
    bb = b.pawns   & b.occupied_co[pl]
    fill_piece(iplanes,5,bb,b,flip_rank,flip_file)

    #black piece attacks
    bb = b.kings   & b.occupied_co[npl]
    fill_piece(iplanes,6,bb,b,flip_rank,flip_file)
    bb = b.queens  & b.occupied_co[npl]
    fill_piece(iplanes,7,bb,b,flip_rank,flip_file)
    bb = b.rooks   & b.occupied_co[npl]
    fill_piece(iplanes,8,bb,b,flip_rank,flip_file)
    bb = b.bishops & b.occupied_co[npl]
    fill_piece(iplanes,9,bb,b,flip_rank,flip_file)
    bb = b.knights & b.occupied_co[npl]
    fill_piece(iplanes,10,bb,b,flip_rank,flip_file)
    bb = b.pawns   & b.occupied_co[npl]
    fill_piece(iplanes,11,bb,b,flip_rank,flip_file)

def fill_planes(iplanes, b):
    """ Compute input planes for ResNet training """

    #fill planes
    flip_rank = (b.turn == chess.BLACK)
    flip_file = (chess.square_file(b.king(b.turn)) < 4)
    fill_planes_(iplanes, b, b.turn, flip_rank, flip_file)

    #enpassant, castling, fifty and on-board mask
    if b.ep_square:
        f = chess.square_file(b.ep_square)
        r = chess.square_rank(b.ep_square)
        if flip_rank: r = RANK_U - r
        if flip_file: f = FILE_U - f
        iplanes[CHANNELS - 8, r, f] = 1

    if b.has_queenside_castling_rights(b.turn):
        iplanes[CHANNELS - (6 if flip_file else 7), :, :] = 1
    if b.has_kingside_castling_rights(b.turn):
        iplanes[CHANNELS - (7 if flip_file else 6), :, :] = 1
    if b.has_queenside_castling_rights(not b.turn):
        iplanes[CHANNELS - (4 if flip_file else 5), :, :] = 1
    if b.has_kingside_castling_rights(not b.turn):
        iplanes[CHANNELS - (5 if flip_file else 4), :, :] = 1

    iplanes[CHANNELS - 3, :, :] = b.fullmove_number / 200.0
    iplanes[CHANNELS - 2, :, :] = b.halfmove_clock / 100.0
    iplanes[CHANNELS - 1, :, :] = 1

def fill_planes_nnue_(iplanes, ikings, b, side):

    #fill planes
    flip_rank = (side == chess.BLACK)
    flip_file = (chess.square_file(b.king(side)) < 4)
    fill_planes_(iplanes, b, side, flip_rank, flip_file)

    #king index
    ksq = b.king(side)
    f = chess.square_file(ksq)
    r = chess.square_rank(ksq)
    if flip_rank: r = RANK_U - r
    if flip_file: f = FILE_U - f
    kindex = r * BOARDX // 2 + (f - BOARDX // 2)
    ikings[0] = NNUE_KINDEX_TAB[NNUE_KIDX][kindex]

def fill_planes_nnue(iplanes, ikings, b):
    """ Compute input planes for NNUE training """

    fill_planes_nnue_(iplanes[:(CHANNELS+NNUE_FACTORIZER_EXTRA),:,:], ikings[:1], b, b.turn)
    fill_planes_nnue_(iplanes[(CHANNELS+NNUE_FACTORIZER_EXTRA):,:,:], ikings[1:], b, not b.turn)

def fill_planes_fen(iplanes, fen, player):

    #flip rank
    flip_rank = (player == 1)

    #board
    kf = 4
    cnt = 0
    for r in range(RANK_U, -1, -1):
        for f in range(0, FILE_U + 1, 1):
            c = fen[cnt]
            idx = PIECE_MAP.find(c)
            if idx != -1:
                if not flip_rank:
                    iplanes[idx, r,  f] = 1
                else:
                    iplanes[(idx^1), RANK_U - r,  f] = 1
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
            temp = iplanes[:, :, FILE_U - f]
            iplanes[:, :, FILE_U - f] = iplanes[:, :, f]
            iplanes[:, :, f] = temp

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
                    if not flip_rank:
                        iplanes[idx   + N_PIECES, :,  :] = holdings[idx] / 50.0
                    else:
                        iplanes[(idx^1) + N_PIECES, :,  :] = holdings[idx] / 50.0
        cnt = cnt + 2

    #enpassant, castling, fifty and on-board mask
    epstr = words[2]
    if epstr[0] != '-':
        f = epstr[0] - 'a'
        r = epstr[1] - '1'
        if flip_rank: r = RANK_U - r
        if flip_file: f = FILE_U - f
        iplanes[CHANNELS - 8, r, f] = 1

    castr = words[1]
    if castr.find('Q' if player == 0 else 'q') != -1:
        iplanes[CHANNELS - (6 if flip_file else 7), :, :] = 1
    if castr.find('K' if player == 0 else 'k') != -1:
        iplanes[CHANNELS - (7 if flip_file else 6), :, :] = 1
    if castr.find('q' if player == 0 else 'Q') != -1:
        iplanes[CHANNELS - (4 if flip_file else 5), :, :] = 1
    if castr.find('k' if player == 0 else 'K') != -1:
        iplanes[CHANNELS - (5 if flip_file else 4), :, :] = 1

    iplanes[CHANNELS - 3, :, :] = float(words[4]) / 200.0
    iplanes[CHANNELS - 2, :, :] = float(words[3]) / 100.0
    iplanes[CHANNELS - 1, :, :] = 1

def fill_examples(examples):

    #arrays
    N = len(examples)
    if HEAD_TYPE == 3:
        iplanes = np.zeros(shape=(N,2*(CHANNELS + NNUE_FACTORIZER_EXTRA),BOARDY,BOARDX),dtype=np.int8)
    else:
        iplanes = np.zeros(shape=(N,CHANNELS,BOARDY,BOARDX),dtype=np.float32)
    oresult = np.zeros(shape=(N,),dtype=np.int)
    if HEAD_TYPE == 0:
        ovalue = np.zeros(shape=(N,3),dtype=np.float32)
        opolicy = np.zeros(shape=(N,BOARDY*BOARDX*POLICY_CHANNELS),dtype=np.float32)
        ret = [iplanes, oresult, ovalue, opolicy]
    elif HEAD_TYPE == 1:
        ovalue = np.zeros(shape=(N,3),dtype=np.float32)
        oscore = np.zeros(shape=(N,BOARDY*BOARDX*POLICY_CHANNELS),dtype=np.float32)
        ret = [iplanes, oresult, ovalue, oscore]
    elif HEAD_TYPE == 2:
        ovalue = np.zeros(shape=(N,3),dtype=np.float32)
        opolicy = np.zeros(shape=(N,BOARDY*BOARDX*POLICY_CHANNELS),dtype=np.float32)
        oscore = np.zeros(shape=(N,BOARDY*BOARDX*POLICY_CHANNELS),dtype=np.float32)
        ret = [iplanes, oresult, ovalue, opolicy, oscore]
    else:
        ovalue = np.zeros(shape=(N,),dtype=np.float32)
        ikings = np.zeros(shape=(N,2),dtype=np.int8)
        ret = [iplanes, oresult, ovalue, ikings]

    #parse each example
    for id,line in enumerate(examples):

        words = line.strip().split()

        #fen string
        fen = ''
        for i in range(0, 6):
            fen = fen + words[i] + ' '

        #player
        if words[1] == 'b':
            player = 1
        else:
            player = 0

        #result
        svalue = words[6]
        if svalue == '1-0':
            result = 0
        elif svalue == '0-1':
            result = 2
        else:
            result = 1

        # value
        value = float(words[7])

        #flip value/result
        if player == 1:
            result = 2 - result
            value = 1 - value

        #result
        oresult[id] = result

        #value/policy/score heads
        if HEAD_TYPE == 3:
            #value
            if FRAC_Z == 0:
                ovalue[id] = value
            elif FRAC_Z == 1:
                ovalue[id] = 1 - result / 2.0
            else:
                ovalue[id] = value * (1 - FRAC_Z) + (1 - result / 2.0) * FRAC_Z
        else:
            # nmoves
            nmoves = int(words[8])
            offset = 9

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
        if HEAD_TYPE == 3:
            bb = chess.Board(fen)
            fill_planes_nnue(iplanes[id,:,:,:],ikings[id,:],bb)
        elif AUX_INP:
            bb = chess.Board(fen)
            fill_planes(iplanes[id,:,:,:],bb)
        else:
            fill_planes_fen(iplanes[id,:,:,:],fen,words,player)

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
        return resnet.build_net(INPUT_SHAPE, 24, 320, POLICY_CHANNELS, HEAD_TYPE)
    elif cid == 5:
        INPUT_SHAPE=(NNUE_CHANNELS, BOARDY, BOARDX)
        return nnue.build_net(INPUT_SHAPE)
    else:
        print("Unsupported network id (Use 0 to 4).")
        sys.exit()

# losses and accuracy
def loss(y_true, y_pred):
    is_legal = tf.greater(y_true, 0)
    y_pred = tf.where(is_legal, y_pred, y_true)
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

def paccuracy(y_true,y_pred):
    is_legal = tf.greater(y_true, 0)
    y_pred = tf.where(is_legal, y_pred, y_true)
    return tf.keras.metrics.categorical_accuracy(y_true, y_pred)

def sloss(y_true, y_pred):
    is_legal = tf.greater(y_true, 0)
    y_pred = tf.where(is_legal, y_pred, y_true)

    su = tf.reduce_sum(tf.cast(is_legal, tf.float32))
    sz = tf.cast(tf.size(is_legal), tf.float32)
    return  (sz / su) * tf.keras.losses.mean_squared_error(y_true, y_pred)

def my_load_model(fname,compile=True):
    return tf.keras.models.load_model(fname, compile=compile,
            custom_objects={'loss': loss,
                            'paccuracy':paccuracy,
                            'sloss':sloss,
                            'clipped_relu':nnue.clipped_relu})

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
                return my_load_model(fname,compile)
        else:
            return my_load_model(fname,compile)

    def compile_model(self,mdx,args):
        if args.opt == 0:
            opt = tf.keras.optimizers.SGD(lr=args.lr, momentum=0.9, nesterov=True)
        else:
            opt = tf.keras.optimizers.Adam(lr=args.lr)

        if args.mixed:
            opt = tf.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(opt)

        #losses and metrics
        if HEAD_TYPE == 0:
            losses = {"value":'categorical_crossentropy', "policya":loss}
            metrics = {"value":'accuracy', "policya":paccuracy}
            loss_weights = [args.val_w, args.pol_w]

        elif HEAD_TYPE == 1:
            losses  = {"value":'categorical_crossentropy', "scorea":sloss}
            metrics = {"value":'accuracy'}
            loss_weights = [args.val_w, args.score_w]

        elif HEAD_TYPE == 2:
            losses  = {"value":'categorical_crossentropy', "policya":loss, "scorea":sloss}
            metrics = {"value":'accuracy', "policya":paccuracy}
            loss_weights = [args.val_w, args.pol_w, args.score_w]
        else:
            losses  = {"value":'mean_squared_error'}
            metrics = {}
            loss_weights = [args.val_w]

        # compile model
        if args.gpus > 1:
            with self.mirrored_strategy.scope():
                mdx.compile(loss=losses,loss_weights=loss_weights,
                      optimizer=opt,metrics=metrics)
        else:
            mdx.compile(loss=losses,loss_weights=loss_weights,
                  optimizer=opt,metrics=metrics)

    def train(self,N,res,local_steps,args):

        # generate X and Y
        nlen = N // args.cores
        slices = [ slice((id*nlen) , (min(N,(id+1)*nlen))) for id in range(args.cores) ]

        if HEAD_TYPE == 3:
            ipln = np.zeros(shape=(N,2*(CHANNELS+NNUE_FACTORIZER_EXTRA),BOARDY,BOARDX),dtype=np.int8)
        else:
            ipln = np.zeros(shape=(N,CHANNELS,BOARDY,BOARDX),dtype=np.float32)
        ores = np.zeros(shape=(N,),dtype=np.int)
        if HEAD_TYPE == 0:
            oval = np.zeros(shape=(N,3),dtype=np.float32)
            opol = np.zeros(shape=(N,BOARDY*BOARDX*POLICY_CHANNELS),dtype=np.float32)
            x = [ipln]
            y = [oval, opol]
        elif HEAD_TYPE == 1:
            oval = np.zeros(shape=(N,3),dtype=np.float32)
            osco = np.zeros(shape=(N,BOARDY*BOARDX*POLICY_CHANNELS),dtype=np.float32)
            x = [ipln]
            y = [oval, osco]
        elif HEAD_TYPE == 2:
            oval = np.zeros(shape=(N,3),dtype=np.float32)
            opol = np.zeros(shape=(N,BOARDY*BOARDX*POLICY_CHANNELS),dtype=np.float32)
            osco = np.zeros(shape=(N,BOARDY*BOARDX*POLICY_CHANNELS),dtype=np.float32)
            x = [ipln]
            y = [oval, opol, osco]
        else:
            oval = np.zeros(shape=(N,),dtype=np.float32)
            ikin = np.zeros(shape=(N,2),dtype=np.int8)
            x1 = np.zeros(shape=(N,NNUE_CHANNELS,BOARDY,BOARDX),dtype=np.int8)
            x2 = np.zeros(shape=(N,NNUE_CHANNELS,BOARDY,BOARDX),dtype=np.int8)
            x = [x1, x2]
            y = [oval]

        #merge results from different cores
        for i in range(args.cores):
            ipln[slices[i],:,:,:] = res[i][0]
            ores[slices[i]] = res[i][1]
            if HEAD_TYPE == 0:
                oval[slices[i],:] = res[i][2]
                opol[slices[i],:] = res[i][3]
            elif HEAD_TYPE == 1:
                oval[slices[i],:] = res[i][2]
                osco[slices[i],:] = res[i][3]
            elif HEAD_TYPE == 2:
                oval[slices[i],:] = res[i][2]
                opol[slices[i],:] = res[i][3]
                osco[slices[i],:] = res[i][4]
            else:
                oval[slices[i]] = res[i][2]
                ikin[slices[i],:] = res[i][3]

        #construct sparse matrix
        if HEAD_TYPE == 3:
            WIDTH=(CHANNELS+NNUE_FACTORIZER_EXTRA)
            for id in range(N):
                if NNUE_KINDICES == 1:
                    x1[id,:,:,:] = ipln[id,:WIDTH,:,:]
                    x2[id,:,:,:] = ipln[id,WIDTH:,:,:]
                else:
                    k1 = ikin[id][0] * CHANNELS
                    k2 = ikin[id][1] * CHANNELS
                    x1[id,k1:k1+CHANNELS,:,:] = ipln[id,:CHANNELS,:,:]
                    x2[id,k2:k2+CHANNELS,:,:] = ipln[id,WIDTH:(WIDTH+CHANNELS),:,:]
                    if NNUE_FACTORIZER != 0:
                        k1 = NNUE_KINDICES * CHANNELS
                        x1[id,k1:k1+CHANNELS,:,:] = ipln[id,:CHANNELS,:,:]
                        x2[id,k1:k1+CHANNELS,:,:] = ipln[id,WIDTH:(WIDTH+CHANNELS),:,:]
                        k1 += CHANNELS
                        if NNUE_FACTORIZER_EXTRA != 0:
                            x1[id,k1:k1+NNUE_FACTORIZER_EXTRA,:,:] = \
                                ipln[id,CHANNELS:WIDTH,:,:]
                            x2[id,k1:k1+NNUE_FACTORIZER_EXTRA,:,:] = \
                                ipln[id,(WIDTH+CHANNELS):,:,:]
                            k1 += NNUE_FACTORIZER_EXTRA

        #sampe weight
        vweights = None
        pweights = None
        if args.pol_grad > 0:
            vweights = np.ones(ores.size)
            pweights = (1.0 - ores / 2.0) - (oval[:,0] + oval[:,1] / 2.0) #  Z-Q

        #train each model
        for i in range(len(self.model)):
            print("Fitting model",i)

            tensorboard_callback = self.cbacks[i]
            initial_epoch = args.global_steps + local_steps
            epochs = initial_epoch + args.epochs

            if args.pol_grad > 0:
                self.model[i].fit(x = x, y = y,
                      batch_size=BATCH_SIZE,
                      sample_weight=[vweights, pweights],
                      validation_split=args.vald_split,
                      initial_epoch=initial_epoch,
                      epochs=epochs,
                      callbacks=[tensorboard_callback])
            else:
                self.model[i].fit(x = x, y = y,
                      batch_size=BATCH_SIZE,
                      validation_split=args.vald_split,
                      initial_epoch=initial_epoch,
                      epochs=epochs,
                      callbacks=[tensorboard_callback])

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

def get_chunk(myNet,args,myEpd,start):

    with (open(myEpd) if not args.gzip else gzip.open(myEpd,mode='rt')) as file:
        count = 0

        examples = []
        start_t = time.time()

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
                chunk = (count + EPD_CHUNK_SIZE - 1) // EPD_CHUNK_SIZE
                end_t = time.time()

                #make sure size is divisible by BATCH_SIZE
                if (not line) and (count % BATCH_SIZE != 0): 
                   count = (count//BATCH_SIZE)*BATCH_SIZE
                   if count > 0:
                        del examples[count:]
                   else:
                        break

                N = len(examples)
                if N > 0:
                    #multiprocess epd
                    nlen = N//args.cores
                    slices = [ slice((id*nlen) , (min(N,(id+1)*nlen))) for id in range(args.cores) ]
                    res = Parallel(n_jobs=args.cores)( delayed(fill_examples) (examples[sl]) for sl in slices )

                    examples = []
                    yield N,res

            #break out
            if not line:
                break

class MyProcess(mp.Process):

    def __init__(self,gen,queue):
        mp.Process.__init__(self)
        self.gen = gen
        self.queue = queue

    def run(self):
        while True:
            if self.queue.qsize() < MAX_QUEUE:
                try:
                    self.queue.put(next(self.gen))
                except StopIteration:
                    break
            else:
                time.sleep(0.1)

def train_epd(myNet,args,myEpd,chunk,start=1):
    gen = get_chunk(myNet,args,myEpd,start)
    s = time.time()
    N,res = next(gen)
    e = time.time()
    print("Average chunk prep time ", int(e - s), "sec")

    queue = mp.Queue()
    p1 = MyProcess(gen,queue)
    p1.start()

    while True:
        print("Training on chunk ", chunk , " ending at position ",
            chunk * EPD_CHUNK_SIZE, " with lr ", args.lr)
        s = time.time()

        myNet.train(N,res,(chunk-1)*NBATCH,args)

        p1.join(timeout=0)
        if p1.is_alive():
            del res
            N,res = queue.get()

        if chunk % args.rsavo == 0:
            myNet.save_checkpoint(chunk, args, True)
        elif chunk % args.rsav == 0:
            myNet.save_checkpoint(chunk, args, False)

        e = time.time()
        print("Total time ", int(e - s), "sec")

        p1.join(timeout=0)
        if not p1.is_alive():
            break

        chunk = chunk + 1

    p1.terminate()
    print("===== Finished training ====")

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
                        help='Nets to train from 0=2x32,6x64,12x128,20x256,4=30x384,5=NNUE.')
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
        help='Heads of neural network, 0=value/policy, 1=value/score, 2=all three, 3=value only.')

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
    exists = os.path.exists(args.dir)
    myNet.save_infer_graph(args)

    #initialize mixed precision training
    if args.mixed:
        config = tf.compat.v1.ConfigProto()
        config.graph_options.rewrite_options.auto_mixed_precision = True
        sess = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(sess)

    #load networks
    print("Loading networks: " + (','.join(str(x) for x in args.nets)) )
    start_t = time.time()
    myNet.load_checkpoint(chunk, args)
    end_t = time.time()
    print("Time", int(end_t - start_t), "sec")

    if args.rand:
        myNet.save_checkpoint(chunk, args)
    else:
        if not exists:
            myNet.save_checkpoint(chunk, args)
        start = chunk * EPD_CHUNK_SIZE + 1
        train_epd(myNet, args, args.epd, chunk, start)


if __name__ == "__main__":
    main(sys.argv[1:])
