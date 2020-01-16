import sys
import os
import logging
import time
import chess
import resnet
import argparse
import gzip
import numpy as np
import multiprocessing
from joblib import Parallel, delayed, dump, load

from keras.models import load_model
from keras import optimizers
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.utils import to_categorical, multi_gpu_model
import tensorflow as tf

AUX_INP = True
CHANNELS = 24
BOARDX = 8
BOARDY = 8
NPOLICY = 4672
NPARMS = 5
NBATCH = 512
BATCH_SIZE = 512
EPD_CHUNK_SIZE = BATCH_SIZE * NBATCH
USE_EPD = False
VALUE_TARGET = 0
PIECE_MAP = "KQRBNPkqrbnp"

def fill_piece(iplanes, ix, bb, b):
    if AUX_INP:
        abb = 0
        squares = chess.SquareSet(bb)
        for sq in squares:
            abb = abb | b.attacks_mask(sq)
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            iplanes[r,  f,  ix + 12] = 1.0

        squares = chess.SquareSet(abb)
        for sq in squares:
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            iplanes[r,  f,  ix] = 1.0
    else:
        squares = chess.SquareSet(bb)
        for sq in squares:
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            iplanes[r,  f,  ix] = 1.0

def fill_planes(iplanes, iparams, b):
    pl = chess.WHITE
    npl = chess.BLACK
    #white piece attacks
    bb = b.kings   & b.occupied_co[pl]
    fill_piece(iplanes,0,bb,b)
    bb = b.queens  & b.occupied_co[pl]
    fill_piece(iplanes,1,bb,b)
    bb = b.rooks   & b.occupied_co[pl]
    fill_piece(iplanes,2,bb,b)
    bb = b.bishops & b.occupied_co[pl]
    fill_piece(iplanes,3,bb,b)
    bb = b.knights & b.occupied_co[pl]
    fill_piece(iplanes,4,bb,b)
    bb = b.pawns   & b.occupied_co[pl]
    fill_piece(iplanes,5,bb,b)
    #black piece attacks
    bb = b.kings   & b.occupied_co[npl]
    fill_piece(iplanes,6,bb,b)
    bb = b.queens  & b.occupied_co[npl]
    fill_piece(iplanes,7,bb,b)
    bb = b.rooks   & b.occupied_co[npl]
    fill_piece(iplanes,8,bb,b)
    bb = b.bishops & b.occupied_co[npl]
    fill_piece(iplanes,9,bb,b)
    bb = b.knights & b.occupied_co[npl]
    fill_piece(iplanes,10,bb,b)
    bb = b.pawns   & b.occupied_co[npl]
    fill_piece(iplanes,11,bb,b)

    #piece counts
    if AUX_INP:
        v = chess.popcount(b.queens  & b.occupied_co[pl]) - chess.popcount(b.queens  & b.occupied_co[npl])
        iparams[0] = v
        v = chess.popcount(b.rooks   & b.occupied_co[pl]) - chess.popcount(b.rooks   & b.occupied_co[npl])
        iparams[1] = v
        v = chess.popcount(b.bishops & b.occupied_co[pl]) - chess.popcount(b.bishops & b.occupied_co[npl])
        iparams[2] = v
        v = chess.popcount(b.knights & b.occupied_co[pl]) - chess.popcount(b.knights & b.occupied_co[npl])
        iparams[3] = v
        v = chess.popcount(b.pawns   & b.occupied_co[pl]) - chess.popcount(b.pawns   & b.occupied_co[npl])
        iparams[4] = v

def fill_planes_fen(iplanes, fen, player):

    RANK_U = BOARDY - 1
    FILE_U = BOARDX - 1

    #board
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

    #holdings
    if fen[cnt - 1] == '[':

        N_PIECES = CHANNELS // 2

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

def fill_examples(examples,iplane,iparam,opolicy,oresult,ovalue):
    if USE_EPD and AUX_INP:
        bb = chess.Board()

    for id,line in enumerate(examples):

        words = line.strip().split()

        epd = ''

        if USE_EPD:
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
            if USE_EPD and AUX_INP:
                bb.set_epd(epd)
            
            #flip board
            if player == 1:
                if USE_EPD and AUX_INP:
                    bb = bb.mirror()
                result = 2 - result
                value = 1 - value

            offset = 9
        else:
            # player
            player = int(words[0])

            # result
            result = int(words[1])

            # value
            value = float(words[2])

            # nmoves
            nmoves = int(words[3])

            #flip board
            if player == 1:
                result = 2 - result
                value = 1 - value

            offset = 4

        #result
        oresult[id] = result

        #value
        if VALUE_TARGET == 0:
            ovalue[id,:] = 0.0
            ovalue[id,result] = 1.0
        else:
            ovalue[id,1] = 0.7 * min(value, 1 - value)
            ovalue[id,0] = value - ovalue[id,1] / 2.0
            ovalue[id,2] = 1 - ovalue[id,0] - ovalue[id,1]
            if VALUE_TARGET == 2:
                ovalue[id,:] /= 2.0
                ovalue[id,result] += 0.5

        #policy
        for i in range(offset, offset+nmoves*2, 2):
            opolicy[id, int(words[i])] = float(words[i+1])

        #input planes
        if USE_EPD:
            if AUX_INP:
                fill_planes(iplane[id,:,:,:],iparam[id,:],bb)
            else:
                fill_planes_fen(iplane[id,:,:,:],epd,player)
        else:
            st=offset+nmoves*2

            if AUX_INP:
                for i in range(0, NPARMS):
                    iparam[id,i] = float(words[st+i])
                st = st + 5

            v1 = int(words[st])
            st = st + 1
            idx = 0
            for i in range(st, len(words)):
                l = int(words[i])
                if v1 > 0:
                    for k in range(0,l):
                        rm = idx % (8 * CHANNELS)
                        r = int(idx // (8 * CHANNELS))
                        s = int(rm // CHANNELS)
                        t = rm % CHANNELS
                        iplane[id,r,s,t] = v1
                        idx = idx + 1
                else:
                    idx = idx + l
                v1 = 1 - v1

def build_model(cid,policy):
    if AUX_INP:
        auxinp = True
    else:
        auxinp = False

    if cid == 0:
        return resnet.build_net((BOARDY, BOARDX, CHANNELS), (NPARMS,),  2,  32, policy, NPOLICY, auxinp)
    elif cid == 1:
        return resnet.build_net((BOARDY, BOARDX, CHANNELS), (NPARMS,),  6,  64, policy, NPOLICY, auxinp)
    elif cid == 2:
        return resnet.build_net((BOARDY, BOARDX, CHANNELS), (NPARMS,), 12, 128, policy, NPOLICY, auxinp)
    elif cid == 3:
        return resnet.build_net((BOARDY, BOARDX, CHANNELS), (NPARMS,), 20, 256, policy, NPOLICY, auxinp)
    elif cid == 4:
        return resnet.build_net((BOARDY, BOARDX, CHANNELS), (NPARMS,), 40, 256, policy, NPOLICY, auxinp)
    else:
        print "Unsupported network id (Use 0 to 4)."
        sys.exit()

class NNet():
    def __init__(self,args):
        self.epochs = args.epochs
        self.lr = args.lr
        self.vald_split = args.vald_split
        self.cores = args.cores
        self.nets = args.nets
        self.rsav = args.rsav
        self.rsavo = args.rsavo
        self.pol_w = args.pol_w
        self.val_w = args.val_w
        self.pol_grad = args.pol_grad

    def new_model(self,cid,args):
        if args.gpus > 1:
            with tf.device('/cpu:0'):
                return build_model(cid, args.policy)
        else:
            return build_model(cid, args.policy)

    def compile_model(self,args):
        if args.opt == 0:
            self.opt = optimizers.SGD(lr=self.lr,momentum=0.9,nesterov=True)
        else:
            self.opt = optimizers.Adam(lr=self.lr)
            
        self.model = []
        for i in range(len(self.cpu_model)):
            if args.gpus > 1:
                self.model.append( multi_gpu_model(self.cpu_model[i], gpus=args.gpus) )
            else:
                self.model.append( self.cpu_model[i] )
            self.model[i].compile(loss=['categorical_crossentropy','categorical_crossentropy'],
                  loss_weights = [self.val_w,self.pol_w],
                  optimizer=self.opt,
                  metrics=['accuracy'])

    def train(self,examples):
        print "Generating input planes using", self.cores, "cores"
        start_t = time.time()

        N = len(examples)

        #memmap
        folder = './joblib_memmap'

        ipln_memmap = os.path.join(folder, 'ipln_memmap')
        ipln = np.memmap(ipln_memmap,dtype=np.float32,shape=(N,BOARDY,BOARDX,CHANNELS),mode='w+')

        ipar_memmap = os.path.join(folder, 'ipar_memmap')
        ipar = np.memmap(ipar_memmap,dtype=np.float32,shape=(N,NPARMS),mode='w+')

        opol_memmap = os.path.join(folder, 'opol_memmap')
        opol = np.memmap(opol_memmap,dtype=np.float32,shape=(N,NPOLICY),mode='w+')

        ores_memmap = os.path.join(folder, 'ores_memmap')
        ores = np.memmap(ores_memmap,dtype=np.int,shape=(N,),mode='w+')

        oval_memmap = os.path.join(folder, 'oval_memmap')
        oval = np.memmap(oval_memmap,dtype=np.float32,shape=(N,3),mode='w+')

        #multiprocess
        nlen = N / self.cores
        slices = [ slice((id*nlen) , (min(N,(id+1)*nlen))) for id in range(self.cores) ]
        Parallel(n_jobs=self.cores)( delayed(fill_examples) (                      \
            examples[sl],ipln[sl,:,:,:],ipar[sl,:],opol[sl,:],ores[sl],oval[sl,:]  \
            ) for sl in slices )

        end_t = time.time()
        print "Time", int(end_t - start_t), "sec"
        
        vweights = None
        pweights = None
        if self.pol_grad > 0:
            vweights = np.ones(ores.size)
            pweights = (1.0 - ores / 2.0) - (oval[:,0] + oval[:,1] / 2.0)

        for i in range(len(self.model)):
            print "Fitting model",i
            if AUX_INP:
                xi = [ipln,ipar]
            else:
                xi = [ipln]

            if self.pol_grad > 0:
                self.model[i].fit(x = xi, y = [oval, opol],
                      batch_size=BATCH_SIZE,
                      sample_weight=[vweights, pweights],
                      validation_split=self.vald_split,
                      epochs=self.epochs)
            else:
                self.model[i].fit(x = xi, y = [oval, opol],
                      batch_size=BATCH_SIZE,
                      validation_split=self.vald_split,
                      epochs=self.epochs)

    def save_checkpoint(self, folder, filename, args, iopt=False):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.mkdir(folder)
        for i,n in enumerate(args.nets):
            fname = filepath  + "-model-" + str(n)
            self.cpu_model[i].save(fname, include_optimizer=iopt)

    def load_checkpoint(self, folder, filename, args):
        filepath = os.path.join(folder, filename)
        self.cpu_model = []
        for n in args.nets:
            fname = filepath  + "-model-" + str(n)
            if not os.path.exists(fname):
                self.cpu_model.append( self.new_model(n,args) )
            else:
                self.cpu_model.append( load_model(fname) )

def train_epd(myNet,args,myEpd,zipped=0,start=1):

    with (open(myEpd) if not zipped else gzip.open(myEpd)) as file:
        count = 0

        examples = []
        start_t = time.time()
        print "Collecting data"

        while True:

            #read fen
            line = file.readline()
            count = count + 1
            if count < start:
                continue

            #train network
            if (not line) or (count % EPD_CHUNK_SIZE == 0):
                if not line:
                   count = count - 1
                   if count % EPD_CHUNK_SIZE == 0:
                      break;

                chunk = (count + EPD_CHUNK_SIZE - 1) / EPD_CHUNK_SIZE
                end_t = time.time()
                if (not line) and (count % BATCH_SIZE != 0): 
                   count = (count//BATCH_SIZE)*BATCH_SIZE
                   if count > 0:
                        del examples[count:]
                   else:
                        break

                print "Time", int(end_t - start_t), "sec"
                print "Training on chunk ", chunk , " ending at position ", count, " with lr ", args.lr
                myNet.train(examples)
                if chunk % myNet.rsavo == 0:
                    myNet.save_checkpoint(args.dir,"ID-" + str(chunk), args, True)
                elif chunk % myNet.rsav == 0:
                    myNet.save_checkpoint(args.dir,"ID-" + str(chunk), args, False)

                examples = []
                start_t = time.time()
                print "Collecting data" 

            #break out
            if not line:
                break

            # add to examples
            examples.append(line)

def main(argv):
    global USE_EPD
    global AUX_INP
    global EPD_CHUNK_SIZE
    global CHANNELS
    global BOARDX
    global BOARDY
    global NPOLICY
    global VALUE_TARGET
    global NBATCH
    global BATCH_SIZE
    global PIECE_MAP

    parser = argparse.ArgumentParser()
    parser.add_argument('--epd','-e', dest='epd', required=False, help='Path to labeled EPD file for training')
    parser.add_argument('--trn','-t', dest='trn', required=False, help='Path to labeled training file')
    parser.add_argument('--dir', dest='dir', required=False, default="nets", help='Path to network files')
    parser.add_argument('--id','-i', dest='id', required=False, type=int, default=0, help='ID of neural network to load.')
    parser.add_argument('--batch-size','-b',dest='batch_size', required=False, type=int, default=BATCH_SIZE, help='Training batch size.')
    parser.add_argument('--nbatch',dest='nbatch', required=False, type=int, default=NBATCH, help='Number of batches to process at one time.')
    parser.add_argument('--epochs',dest='epochs', required=False, type=int, default=1, help='Training epochs.')
    parser.add_argument('--learning-rate','-l',dest='lr', required=False, type=float, default=0.01, help='Training learning rate.')
    parser.add_argument('--vald-split',dest='vald_split', required=False, type=float, default=0.0, help='Fraction of sample to use for validation.')
    parser.add_argument('--cores',dest='cores', required=False, type=int, default=multiprocessing.cpu_count(), help='Number of cores to use.')
    parser.add_argument('--gpus',dest='gpus', required=False, type=int, default=0, help='Number of gpus to use.')
    parser.add_argument('--gzip','-z',dest='gzip', required=False, action='store_true',help='Process zipped file.')
    parser.add_argument('--nets',dest='nets', nargs='+', required=False, type=int, default=[0,1,2], \
                        help='Nets to train from 0=2x32,6x64,12x128,20x256,4=40x256.')
    parser.add_argument('--rsav',dest='rsav', required=False, type=int, default=1, help='Save graph every RSAV chunks.')
    parser.add_argument('--rsavo',dest='rsavo', required=False, type=int, default=20, help='Save optimization state every RSAVO chunks.')
    parser.add_argument('--rand',dest='rand', required=False, action='store_true', help='Generate random network.')
    parser.add_argument('--opt',dest='opt', required=False, type=int, default=0, help='Optimizer 0=SGD 1=Adam.')
    parser.add_argument('--pol',dest='policy', required=False, type=int,default=1, help='Policy head style 0=simple, 1=A0 style')
    parser.add_argument('--pol_w',dest='pol_w', required=False, type=float, default=1.0, help='Policy loss weight.')
    parser.add_argument('--val_w',dest='val_w', required=False, type=float, default=1.0, help='Value loss weight.')
    parser.add_argument('--pol_grad',dest='pol_grad', required=False, type=int, default=0, help='0=standard 1=multiply policy by score.')
    parser.add_argument('--noauxinp','-u',dest='noauxinp', required=False, action='store_false', help='Don\'t use auxillary input')
    parser.add_argument('--channels','-c', dest='channels', required=False, type=int, default=CHANNELS, help='number of input channels of network.')
    parser.add_argument('--boardx','-x', dest='boardx', required=False, type=int, default=BOARDX, help='board x-dimension.')
    parser.add_argument('--boardy','-y', dest='boardy', required=False, type=int, default=BOARDY, help='board y-dimension.')
    parser.add_argument('--npolicy', dest='npolicy', required=False, type=int, default=NPOLICY, help='The number of maximum possible moves.')
    parser.add_argument('--value-target',dest='value_target', required=False, type=int, default=VALUE_TARGET, help='Value target 0=z, 1=q and 2=(q+z)/2.')
    parser.add_argument('--piece-map',dest='pcmap', required=False, default=PIECE_MAP,help='Map pieces to planes')

    args = parser.parse_args()

    # allow memory growth
    logging.getLogger('tensorflow').disabled=True
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)

    # init net
    myNet = NNet(args)

    NBATCH = args.nbatch
    BATCH_SIZE = args.batch_size
    EPD_CHUNK_SIZE = BATCH_SIZE * NBATCH
    CHANNELS = args.channels
    BOARDX = args.boardx
    BOARDY = args.boardy
    NPOLICY = args.npolicy
    AUX_INP = args.noauxinp
    VALUE_TARGET = args.value_target
    PIECE_MAP = args.pcmap

    chunk = args.id

    start_t = time.time()
    print "Loadng networks"
    myNet.load_checkpoint(args.dir,"ID-" + str(chunk), args)
    myNet.compile_model(args)
    end_t = time.time()
    print "Time", int(end_t - start_t), "sec"

    if args.rand:
        myNet.save_checkpoint(args.dir,"ID-" + str(chunk), args, False)
    elif (args.epd != None) or (args.trn != None):
        folder = './joblib_memmap'
        if not os.path.isdir(folder):
            os.mkdir(folder)

        start = chunk * EPD_CHUNK_SIZE + 1
        if args.epd != None:
            USE_EPD = True
            train_epd(myNet, args, args.epd, args.gzip, start)
        else:
            USE_EPD = False
            train_epd(myNet, args, args.trn, args.gzip, start)


if __name__ == "__main__":
    main(sys.argv[1:])
