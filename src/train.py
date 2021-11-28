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
from ctypes import cdll, c_int, c_char_p, POINTER, pointer

#import tensorflow and set logging level
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except:
    pass
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

#global params
AUX_INP = True
CHANNELS = 32
BOARDX = 8
BOARDY = 8
POLICY_CHANNELS = 16
BATCH_SIZE = 512
PIECE_MAP = "KQRBNPkqrbnp"
RANK_U = BOARDY - 1
FILE_U = BOARDX - 1
FRAC_PI = 1
FRAC_Z  = 1
HEAD_TYPE = 0
MAX_QUEUE = 1
use_data_loader = True
data_loader = None

#NNUE
NNUE_KIDX = 4
NNUE_KINDICES = (1 << NNUE_KIDX)
NNUE_FACTORIZER = (12 if NNUE_KIDX > 0 else 0)
NNUE_FACTORIZER_EXTRA = 2
NNUE_CHANNELS = NNUE_KINDICES * 12 + NNUE_FACTORIZER + NNUE_FACTORIZER_EXTRA
NNUE_FEATURES = NNUE_CHANNELS * BOARDY * BOARDX

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

#NN
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
            iplanes[r,  f,  ix + 12] = 1.0

        squares = chess.SquareSet(abb)
        for sq in squares:
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            if flip_rank: r = RANK_U - r
            if flip_file: f = FILE_U - f
            iplanes[r,  f,  ix] = 1.0
    else:
        squares = chess.SquareSet(bb)
        for sq in squares:
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            if flip_rank: r = RANK_U - r
            if flip_file: f = FILE_U - f
            iplanes[r,  f,  ix] = 1.0

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
        iplanes[r, f, CHANNELS - 8] = 1.0

    if b.has_queenside_castling_rights(b.turn):
        iplanes[:, :, CHANNELS - (6 if flip_file else 7)] = 1.0
    if b.has_kingside_castling_rights(b.turn):
        iplanes[:, :, CHANNELS - (7 if flip_file else 6)] = 1.0
    if b.has_queenside_castling_rights(not b.turn):
        iplanes[:, :, CHANNELS - (4 if flip_file else 5)] = 1.0
    if b.has_kingside_castling_rights(not b.turn):
        iplanes[:, :, CHANNELS - (5 if flip_file else 4)] = 1.0

    iplanes[:, :, CHANNELS - 3] = b.fullmove_number / 200.0
    iplanes[:, :, CHANNELS - 2] = b.halfmove_clock / 100.0
    iplanes[:, :, CHANNELS - 1] = 1.0

#NNUE

def fill_piece_nnue(iplanes, cidx, kidx, ix, bb, flip_rank, flip_file):
    """ Compute piece placement and attack plane for a given piece type """
    squares = chess.SquareSet(bb)
    for sq in squares:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        if flip_rank: r = RANK_U - r
        if flip_file: f = FILE_U - f
        off = (r * 8 + f) * NNUE_CHANNELS + ix

        iplanes[cidx,1] = kidx * 12 + off
        cidx = cidx + 1

        if NNUE_FACTORIZER > 0:
            iplanes[cidx,1] = NNUE_KINDICES * 12 + off
            cidx = cidx + 1

    return cidx

def flip_vertical(bb):
    # https://www.chessprogramming.org/Flipping_Mirroring_and_Rotating#FlipVertically
    bb = ((bb >> 8) & 0x00ff00ff00ff00ff) | ((bb & 0x00ff00ff00ff00ff) << 8)
    bb = ((bb >> 16) & 0x0000ffff0000ffff) | ((bb & 0x0000ffff0000ffff) << 16)
    bb = (bb >> 32) | ((bb & 0x00000000ffffffff) << 32)
    return bb

def flip_horizontal(bb):
    # https://www.chessprogramming.org/Flipping_Mirroring_and_Rotating#MirrorHorizontally
    bb = ((bb >> 1) & 0x5555555555555555) | ((bb & 0x5555555555555555) << 1)
    bb = ((bb >> 2) & 0x3333333333333333) | ((bb & 0x3333333333333333) << 2)
    bb = ((bb >> 4) & 0x0f0f0f0f0f0f0f0f) | ((bb & 0x0f0f0f0f0f0f0f0f) << 4)
    return bb

def fill_planes_nnue_(iplanes, ivalues, cidx, pid, kidx, b, side, flip_rank, flip_file):
    """ Compute piece and attack planes for all pieces"""

    pl = side
    npl = not side

    st = cidx

    #white piece attacks
    bb = b.kings   & b.occupied_co[pl]
    cidx = fill_piece_nnue(iplanes,cidx,kidx,0,bb,flip_rank,flip_file)
    bb = b.queens  & b.occupied_co[pl]
    cidx = fill_piece_nnue(iplanes,cidx,kidx,1,bb,flip_rank,flip_file)
    bb = b.rooks   & b.occupied_co[pl]
    cidx = fill_piece_nnue(iplanes,cidx,kidx,2,bb,flip_rank,flip_file)
    bb = b.bishops & b.occupied_co[pl]
    cidx = fill_piece_nnue(iplanes,cidx,kidx,3,bb,flip_rank,flip_file)
    bb = b.knights & b.occupied_co[pl]
    cidx = fill_piece_nnue(iplanes,cidx,kidx,4,bb,flip_rank,flip_file)
    bb = b.pawns   & b.occupied_co[pl]
    cidx = fill_piece_nnue(iplanes,cidx,kidx,5,bb,flip_rank,flip_file)

    #black piece attacks
    bb = b.kings   & b.occupied_co[npl]
    cidx = fill_piece_nnue(iplanes,cidx,kidx,6,bb,flip_rank,flip_file)
    bb = b.queens  & b.occupied_co[npl]
    cidx = fill_piece_nnue(iplanes,cidx,kidx,7,bb,flip_rank,flip_file)
    bb = b.rooks   & b.occupied_co[npl]
    cidx = fill_piece_nnue(iplanes,cidx,kidx,8,bb,flip_rank,flip_file)
    bb = b.bishops & b.occupied_co[npl]
    cidx = fill_piece_nnue(iplanes,cidx,kidx,9,bb,flip_rank,flip_file)
    bb = b.knights & b.occupied_co[npl]
    cidx = fill_piece_nnue(iplanes,cidx,kidx,10,bb,flip_rank,flip_file)
    bb = b.pawns   & b.occupied_co[npl]
    cidx = fill_piece_nnue(iplanes,cidx,kidx,11,bb,flip_rank,flip_file)

    ivalues[st:cidx] = 1

    #extra factorizers
    if NNUE_FACTORIZER_EXTRA > 0:

        OFF0 = (NNUE_KINDICES + 1) * 12
        OFF1 = OFF0 + 1
      
        for ix in range(0,6):
            bb = b.pieces_mask(chess.PIECE_TYPES[5 - ix], pl)
            if flip_file: bb = flip_horizontal(bb)
            if flip_rank: bb = flip_vertical(bb)

            #file
            for f in range(0,8):
                cnt = chess.popcount(bb & chess.BB_FILES[f])
                if cnt > 0:
                    iplanes[cidx,1] = (ix * 8 + f) * NNUE_CHANNELS  + OFF0
                    ivalues[cidx] = cnt
                    cidx = cidx + 1

            #rank
            for r in range(0,8):
                cnt = chess.popcount(bb & chess.BB_RANKS[r])
                if cnt > 0:
                    iplanes[cidx,1] = (ix * 8 + r) * NNUE_CHANNELS  + OFF1
                    ivalues[cidx] = cnt
                    cidx = cidx + 1

            #four rings
            BB_RING_0 = 0xffffffffffffffff
            BB_RING_1 = 0x007e7e7e7e7e7e00
            BB_RING_2 = 0x00003c3c3c3c0000
            BB_RING_3 = 0x0000001818000000

            cnt = chess.popcount(bb & BB_RING_0)
            if cnt > 0:
                iplanes[cidx,1] = (6 * 8 + ix) * NNUE_CHANNELS  + OFF0
                ivalues[cidx] = cnt
                cidx = cidx + 1

                cnt = chess.popcount(bb & BB_RING_1)
                if cnt > 0:
                    iplanes[cidx,1] = (7 * 8 + ix) * NNUE_CHANNELS  + OFF0
                    ivalues[cidx] = cnt
                    cidx = cidx + 1

                    cnt = chess.popcount(bb & BB_RING_2)
                    if cnt > 0:
                        iplanes[cidx,1] = (6 * 8 + ix) * NNUE_CHANNELS  + OFF1
                        ivalues[cidx] = cnt
                        cidx = cidx + 1

                        cnt = chess.popcount(bb & BB_RING_3)
                        if cnt > 0:
                            iplanes[cidx,1] = (7 * 8 + ix) * NNUE_CHANNELS  + OFF1
                            ivalues[cidx] = cnt
                            cidx = cidx + 1

            #dark squares
            if ix == 3:
                cnt = chess.popcount(bb & chess.BB_DARK_SQUARES)
                if cnt > 0:
                    iplanes[cidx,1] = (6 * 8 + 6) * NNUE_CHANNELS  + OFF0
                    ivalues[cidx] = cnt
                    cidx = cidx + 1
            if ix == 4:
                cnt = chess.popcount(bb & chess.BB_DARK_SQUARES)
                if cnt > 0:
                    iplanes[cidx,1] = (6 * 8 + 7) * NNUE_CHANNELS  + OFF0
                    ivalues[cidx] = cnt
                    cidx = cidx + 1
            if ix == 5:
                cnt = chess.popcount(bb & chess.BB_DARK_SQUARES)
                if cnt > 0:
                    iplanes[cidx,1] = (7 * 8 + 6) * NNUE_CHANNELS  + OFF0
                    ivalues[cidx] = cnt
                    cidx = cidx + 1

                    cnt = chess.popcount(bb & chess.BB_DARK_SQUARES & BB_RING_2)
                    if cnt > 0:
                        iplanes[cidx,1] = (7 * 8 + 7) * NNUE_CHANNELS  + OFF0
                        ivalues[cidx] = cnt
                        cidx = cidx + 1

            #diagonals
            BB_DIAG_A1H8  = 0x8040201008040201
            BB_DIAG_A8H1  = 0x0102040810204080
            BB_DIAG_A1H8N = 0x40a05028140a0502
            BB_DIAG_A8H1N = 0x02050a142850a040

            if ix == 3:
                cnt = chess.popcount(bb & BB_DIAG_A1H8)
                if cnt > 0:
                    iplanes[cidx,1] = (6 * 8 + 6) * NNUE_CHANNELS  + OFF1
                    ivalues[cidx] = cnt
                    cidx = cidx + 1
                cnt = chess.popcount(bb & BB_DIAG_A8H1)
                if cnt > 0:
                    iplanes[cidx,1] = (6 * 8 + 7) * NNUE_CHANNELS  + OFF1
                    ivalues[cidx] = cnt
                    cidx = cidx + 1
                cnt = chess.popcount(bb & BB_DIAG_A1H8N)
                if cnt > 0:
                    iplanes[cidx,1] = (7 * 8 + 6) * NNUE_CHANNELS  + OFF1
                    ivalues[cidx] = cnt
                    cidx = cidx + 1
                cnt = chess.popcount(bb & BB_DIAG_A8H1N)
                if cnt > 0:
                    iplanes[cidx,1] = (7 * 8 + 7) * NNUE_CHANNELS  + OFF1
                    ivalues[cidx] = cnt
                    cidx = cidx + 1

    iplanes[st:cidx, 0] = pid

    return cidx

def fill_planes_nnue_one(iplanes, ivalues, cidx, pid,  b, side):

    #king index
    ksq = b.king(side)
    f = chess.square_file(ksq)
    r = chess.square_rank(ksq)
    flip_rank = (side == chess.BLACK)
    flip_file = (f < 4)
    if flip_rank: r = RANK_U - r
    if flip_file: f = FILE_U - f
    kindex = r * BOARDX // 2 + (f - BOARDX // 2)
    kidx = NNUE_KINDEX_TAB[NNUE_KIDX][kindex]

    #fill planes
    return fill_planes_nnue_(iplanes, ivalues, cidx, pid, kidx, b, side, flip_rank, flip_file)

def fill_planes_nnue(iplanes, ivalues, cidx0, cidx1, pid, b):
    """ Compute input planes for NNUE training """

    cidx0 = fill_planes_nnue_one(iplanes[0,:,:], ivalues[0,:], cidx0, pid, b, b.turn)
    cidx1 = fill_planes_nnue_one(iplanes[1,:,:], ivalues[1,:], cidx1, pid, b, not b.turn)

    return cidx0,cidx1

# FEN for variants
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
                    if not flip_rank:
                        iplanes[:,  :,    idx   + N_PIECES] = holdings[idx] / 50.0
                    else:
                        iplanes[:,  :,  (idx^1) + N_PIECES] = holdings[idx] / 50.0
        cnt = cnt + 2

    #enpassant, castling, fifty and on-board mask
    epstr = words[2]
    if epstr[0] != '-':
        f = epstr[0] - 'a'
        r = epstr[1] - '1'
        if flip_rank: r = RANK_U - r
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

def fill_examples(examples, spid):

    #arrays
    N = len(examples)
    if HEAD_TYPE == 3:
        iplanes = np.zeros(shape=(2,N*192,2), dtype=np.int32)
        ivalues = np.zeros(shape=(2,N*192), dtype=np.int8)
    else:
        iplanes = np.zeros(shape=(N,BOARDY,BOARDX,CHANNELS),dtype=np.float32)
    oresult = np.zeros(shape=(N,),dtype=np.int8)
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

    #parse each example
    cidx0,cidx1 = 0,0

    for id,line in enumerate(examples):

        words = line.strip().split()

        #fen string
        fen = " ".join(words[0:6])

        #player
        if words[1] == 'b':
            player = 1
        else:
            player = 0

        #result
        if words[6] == '1-0':
            result = 0
        elif words[6] == '0-1':
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
            cidx0,cidx1 = fill_planes_nnue(iplanes,ivalues,cidx0,cidx1,spid+id,bb)
        elif AUX_INP:
            bb = chess.Board(fen)
            fill_planes(iplanes[id,:,:,:],bb)
        else:
            fill_planes_fen(iplanes[id,:,:,:],fen,words,player)

    if HEAD_TYPE == 3:
        ret = [iplanes[0,:cidx0,:], iplanes[1,:cidx1,:], ivalues[0,:cidx0], ivalues[1,:cidx1], oresult, ovalue]

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
        INPUT_SHAPE=(NNUE_FEATURES,)
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
                            'clipped_relu':nnue.clipped_relu,
                            'DenseLayerForSparse':nnue.DenseLayerForSparse})

class NNet():
    def __init__(self,args):
        self.mirrored_strategy = tf.distribute.MirroredStrategy()

    def new_model(self,args):
        if args.gpus > 1:
            with self.mirrored_strategy.scope():
                return build_model(args.net)
        else:
            return build_model(args.net)

    def load_model(self,fname,compile,args):
        if args.gpus > 1:
            with self.mirrored_strategy.scope():
                return my_load_model(fname,compile)
        else:
            return my_load_model(fname,compile)

    def compile_model(self,mdx,args):
        if args.opt == 0:
            opt = tf.keras.optimizers.SGD(learning_rate=args.lr, momentum=0.9, nesterov=True)
        else:
            opt = tf.keras.optimizers.Adam(learning_rate=args.lr)

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

    def train(self,gen,local_steps,args):

        #initial steps
        initial_steps = args.global_steps + local_steps

        for steps in range(args.max_steps):
            t_steps = steps + initial_steps

            self.callbacks.on_train_batch_begin(steps)

            x,y = next(gen)
            logs = self.model.train_on_batch(x,y,reset_metrics = False,return_dict = True)

            self.callbacks.on_train_batch_end(steps, logs)

            self.tensorboard.on_epoch_end(t_steps, logs)

            nid = (steps + 1) // args.rsav
            if (steps + 1) % args.rsavo == 0:
                self.save_checkpoint(nid, args, True)
            elif (steps + 1) % args.rsav == 0:
                self.save_checkpoint(nid, args, False)

    def save_checkpoint(self, nid, args, iopt=False):
        filepath = os.path.join(args.dir, "ID-" + str(nid))
        if not os.path.exists(args.dir):
            os.mkdir(args.dir)

        #save each model
        fname = filepath  + "-model-" + str(args.net)
        self.model.save(fname, include_optimizer=iopt, save_format='h5')

    def load_checkpoint(self, nid, args):
        filepath = os.path.join(args.dir, "ID-" + str(nid))

        #create training model]
        fname = filepath  + "-model-" + str(args.net)
        if not os.path.exists(fname):
            mdx = self.new_model(args)
            self.compile_model(mdx, args)
        else:
            comp = ((nid * args.rsav) % args.rsavo == 0)
            mdx = self.load_model(fname,comp,args)
            if not mdx.optimizer:
                print("====== ", fname, " : starting from fresh optimizer state ======")
                self.compile_model(mdx, args)
        self.model = mdx

        #common callbacks list
        self.callbacks = tf.keras.callbacks.CallbackList(
            None,
            add_history = True,
            add_progbar = True,
            model = self.model,
            epochs = 1,
            verbose = 1,
            steps = args.max_steps
        )

        #tensorboard callback
        log_dir = "logs/fit/model-" + str(args.net)
        self.tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            update_freq=args.rsav
        )
        self.tensorboard.set_model(self.model)

    def save_infer_graph(self, args):
        filepath = os.path.join(args.dir, "infer-")
        if not os.path.exists(args.dir):
            os.mkdir(args.dir)

        tf.keras.backend.set_learning_phase(0)

        #create inference model
        fname = filepath + str(args.net)
        new_model = self.new_model(args)
        new_model.save(fname, include_optimizer=False, save_format='h5')

        tf.keras.backend.clear_session()
        tf.keras.backend.set_learning_phase(1)

def prep_data(N,examples,args):

    #multiprocess epd
    nlen = int(round(N/args.cores))
    slices = [ slice((id*nlen) , (min(N,(id+1)*nlen))) for id in range(args.cores) ]
    res = Parallel(n_jobs=args.cores)( delayed(fill_examples) (examples[sl],sl.start) for sl in slices )

    # generate X and Y
    if HEAD_TYPE == 3:
        S1,S2 = 0,0
        for i in range(args.cores):
            S1 = S1 + res[i][0].shape[0]
            S2 = S2 + res[i][1].shape[0]
        iplanes0 = np.zeros(shape=(S1,2), dtype=np.int32)
        iplanes1 = np.zeros(shape=(S2,2), dtype=np.int32)
        ivalues0 = np.zeros(shape=(S1), dtype=np.int8)
        ivalues1 = np.zeros(shape=(S2), dtype=np.int8)
        x = (iplanes0, iplanes1, ivalues0, ivalues1)
    else:
        ipln = np.zeros(shape=(N,CHANNELS,BOARDY,BOARDX),dtype=np.float32)
    ores = np.zeros(shape=(N,),dtype=np.int8)
    if HEAD_TYPE == 0:
        oval = np.zeros(shape=(N,3),dtype=np.float32)
        opol = np.zeros(shape=(N,BOARDY*BOARDX*POLICY_CHANNELS),dtype=np.float32)
        x = (ipln)
        y = (oval, opol)
    elif HEAD_TYPE == 1:
        oval = np.zeros(shape=(N,3),dtype=np.float32)
        osco = np.zeros(shape=(N,BOARDY*BOARDX*POLICY_CHANNELS),dtype=np.float32)
        x = (ipln)
        y = (oval, osco)
    elif HEAD_TYPE == 2:
        oval = np.zeros(shape=(N,3),dtype=np.float32)
        opol = np.zeros(shape=(N,BOARDY*BOARDX*POLICY_CHANNELS),dtype=np.float32)
        osco = np.zeros(shape=(N,BOARDY*BOARDX*POLICY_CHANNELS),dtype=np.float32)
        x = (ipln)
        y = (oval, opol, osco)
    else:
        oval = np.zeros(shape=(N,),dtype=np.float32)
        y = (oval)

    #merge results from different cores
    cidx0,cidx1 = 0,0
    for i in range(args.cores):
        if HEAD_TYPE == 3:
            S1 = res[i][0].shape[0]
            S2 = res[i][1].shape[0]
            iplanes0[cidx0:(cidx0 + S1),:] = res[i][0]
            iplanes1[cidx1:(cidx1 + S2),:] = res[i][1]
            ivalues0[cidx0:(cidx0 + S1)] = res[i][2]
            ivalues1[cidx1:(cidx1 + S2)] = res[i][3]
            cidx0 = cidx0 + S1
            cidx1 = cidx1 + S2
            ores[slices[i]] = res[i][4]
        else:
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
            oval[slices[i]] = res[i][5]

    #construct sparse matrix
    if HEAD_TYPE == 3:
        dense_shape = (N,NNUE_FEATURES)
        x1 = tf.sparse.reorder(tf.SparseTensor(x[0],x[2],dense_shape))
        x2 = tf.sparse.reorder(tf.SparseTensor(x[1],x[3],dense_shape))
        x = (x1, x2)

    return x,y

def prep_data_nnue(N,examples,args):
    global data_loader

    iplanes0 = np.zeros(shape=(N*192,2), dtype=np.int32)
    ivalues0 = np.zeros(shape=(N*192), dtype=np.int32)
    iplanes1 = np.zeros(shape=(N*192,2), dtype=np.int32)
    ivalues1 = np.zeros(shape=(N*192), dtype=np.int32)
    oval = np.zeros(shape=(N,),dtype=np.float32)
    cidx0 = c_int()
    cidx1 = c_int()

    epd = ''.join(examples)
    
    data_loader.generate_input_nnue(
        epd.encode(),
        iplanes0,ivalues0,
        iplanes1,ivalues1,oval,
        pointer(cidx0),pointer(cidx1))

    cidx0 = cidx0.value
    cidx1 = cidx1.value

    x = (iplanes0[:cidx0,:], iplanes1[:cidx1,:], ivalues0[:cidx0], ivalues1[:cidx1])
    y = (oval)

    #construct sparse matrix
    if HEAD_TYPE == 3:
        dense_shape = (N,NNUE_FEATURES)
        x1 = tf.sparse.reorder(tf.SparseTensor(x[0],x[2],dense_shape))
        x2 = tf.sparse.reorder(tf.SparseTensor(x[1],x[3],dense_shape))
        x = (x1, x2)

    return x,y

def get_batch(myNet,args,myEpd,start):

    with (open(myEpd) if not args.gzip else gzip.open(myEpd,mode='rt')) as file:
        count = 0

        examples = []

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
            if (not line) or (count % BATCH_SIZE == 0):

                #make sure size is divisible by BATCH_SIZE
                if (not line) and (count % BATCH_SIZE != 0):
                    count = (count//BATCH_SIZE)*BATCH_SIZE
                    if count > 0:
                        del examples[count:]
                    else:
                        break

                #prep ML data
                N = len(examples)
                if N > 0:
                    if use_data_loader:
                        x,y = prep_data_nnue(N,examples,args)
                    else:
                        x,y = prep_data(N,examples,args)
                    examples = []
                    yield x,y

            #break out
            if not line:
                break

def train_epd(myNet,args,myEpd,nid,start=1):
    gen = get_batch(myNet,args,myEpd,start)
    myNet.train(gen,nid*args.rsav,args)

def main(argv):
    global AUX_INP, CHANNELS, BOARDX, BOARDY, FRAC_Z, FRAC_PI
    global POLICY_CHANNELS,  BATCH_SIZE, PIECE_MAP, RANK_U, FILE_U, HEAD_TYPE

    if use_data_loader:
        global data_loader
        data_loader = cdll.LoadLibrary("data_loader/data_loader.so")
        data_loader.generate_input_nnue.restype = None
        data_loader.generate_input_nnue.argtypes = [
            c_char_p,
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=2, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=2, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
            POINTER(c_int),
            POINTER(c_int) ]

    parser = argparse.ArgumentParser()
    parser.add_argument('--epd','-e', dest='epd', required=False, help='Path to labeled EPD file for training')
    parser.add_argument('--dir', dest='dir', required=False, default="nets", help='Path to network files')
    parser.add_argument('--id','-i', dest='id', required=False, type=int, default=0, help='ID of neural networks to load.')
    parser.add_argument('--global-steps', dest='global_steps', required=False, type=int, default=0, help='Global number of steps trained so far.')
    parser.add_argument('--batch-size','-b',dest='batch_size', required=False, type=int, default=BATCH_SIZE, help='Training batch size.')
    parser.add_argument('--learning-rate','-l',dest='lr', required=False, type=float, default=0.01, help='Training learning rate.')
    parser.add_argument('--validation-split',dest='vald_split', required=False, type=float, default=0.125, help='Fraction of sample to use for validation.')
    parser.add_argument('--cores',dest='cores', required=False, type=int, default=mp.cpu_count(), help='Number of cores to use.')
    parser.add_argument('--gpus',dest='gpus', required=False, type=int, default=0, help='Number of gpus to use.')
    parser.add_argument('--gzip','-z',dest='gzip', required=False, action='store_true',help='Process zipped file.')
    parser.add_argument('--net',dest='net', required=False, type=int, default=0, \
                        help='Net to train from 0=2x32,6x64,12x128,20x256,4=30x384,5=NNUE.')
    parser.add_argument('--rsav',dest='rsav', required=False, type=int, default=16, help='Save graph every RSAV steps.')
    parser.add_argument('--rsavo',dest='rsavo', required=False, type=int, default=128, help='Save optimization state every RSAVO steps.')
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
    parser.add_argument('--max-steps',dest='max_steps', required=False, type=int, default=1000000, \
        help='Maximum number of steps to train for.')

    args = parser.parse_args()

    tf.keras.backend.set_learning_phase(1)

    #memory growth of gpus
    if args.gpus:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        print("Num GPUs:", len(gpus))
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # init net
    myNet = NNet(args)

    BATCH_SIZE = args.batch_size
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

    nid = args.id

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
    print("Loading network: " + str(args.net))
    myNet.load_checkpoint(nid, args)

    if args.rand:
        myNet.save_checkpoint(nid, args, True)
    else:
        if not exists:
            myNet.save_checkpoint(nid, args, True)
        start = nid * args.rsav * BATCH_SIZE + 1
        train_epd(myNet, args, args.epd, nid, start)

if __name__ == "__main__":
    main(sys.argv[1:])
