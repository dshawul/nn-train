import sys
import os
import time
import chess
import chess.pgn
import resnet
import argparse
import gzip
import numpy as np
import multiprocessing
from joblib import Parallel, delayed

from keras.models import load_model
from keras import optimizers
from keras.utils.training_utils import multi_gpu_model
from keras import backend as K
from keras.utils import np_utils
import tensorflow as tf

CHANNELS = 24
NPARMS = 5
EPD_CHUNK_SIZE = 4096 * 80
USE_EPD = False

def fill_piece(iplanes, ix, bb, b):
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

def fill_examples(examples, polt):

    global USE_EPD

    exams = []
    if USE_EPD:
        bb = chess.Board()

    for _,line in enumerate(examples):

        words = line.strip().split()

        if USE_EPD:
            epd = ''
            for i in range(0, 6):
                epd = epd + words[i] + ' '

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
            bb.set_epd(epd)
            
            #flip board
            if bb.turn == chess.BLACK:
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
        iresult = result

        #value
        ivalue = value

        # parse move
        if polt == 0:
            NPOLICY = 1858
        else:
            NPOLICY = 4672

        ipolicy = np.zeros(shape=(NPOLICY,),dtype=np.float32)
        for i in range(offset, offset+nmoves*2, 2):
            ipolicy[int(words[i])] = float(words[i+1])

        #input planes
        iparam = np.zeros(shape=(NPARMS),dtype=np.float32)
        iplane = None
        
        if USE_EPD:
            iplane = np.zeros(shape=(8,8,CHANNELS),dtype=np.float32)
            fill_planes(iplane,iparam,bb)
        else:
            iplane = np.zeros(shape=(8*8*CHANNELS),dtype=np.float32)
            st=offset+nmoves*2
            for i in range(0, NPARMS):
                iparam[i] = float(words[st+i])

            st = st + 5
            v1 = int(words[st])
            st = st + 1
            idx = 0
            for i in range(st, len(words)):
                l = int(words[i])
                if v1 > 0:
                    for k in range(0,l):
                        iplane[idx] = v1
                        idx = idx + 1
                else:
                    idx = idx + l
                v1 = 1 - v1

            iplane = np.reshape(iplane,(8,8,CHANNELS))

        #append
        exams.append([iplane,iparam,iresult,ivalue,ipolicy])

    return exams

def build_model(cid,policy):

    if cid == 0:
        return resnet.build_net((8, 8, CHANNELS), (NPARMS,),  2,  32, policy)
    elif cid == 1:
        return resnet.build_net((8, 8, CHANNELS), (NPARMS,),  6,  64, policy)
    elif cid == 2:
        return resnet.build_net((8, 8, CHANNELS), (NPARMS,), 12, 128, policy)
    elif cid == 3:
        return resnet.build_net((8, 8, CHANNELS), (NPARMS,), 20, 256, policy)
    elif cid == 4:
        return resnet.build_net((8, 8, CHANNELS), (NPARMS,), 40, 256, policy)
    else:
        print "Unsupported network id (Use 0 to 4)."
        sys.exit()

class NNet():
    def __init__(self,args):
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.lr = args.lr
        self.vald_split = args.vald_split
        self.cores = args.cores
        self.nets = args.nets
        self.rsav = args.rsav
        self.rsavo = args.rsavo
        self.pol = args.policy
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
            self.opt = optimizers.SGD(lr=self.lr)
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

        nsz = len(examples)
        nlen = nsz / self.cores
        
        res = Parallel(n_jobs=self.cores)(delayed(fill_examples)\
            ( examples[ (id*nlen) : (min(nsz,(id+1)*nlen)) ], self.pol ) for id in range(self.cores))
        exams = []
        for i in range(self.cores):
            exams = exams + res[i]

        end_t = time.time()
        print "Time", int(end_t - start_t), "sec"
        
        ipln, ipar, ores, oval, opol = list(zip(*exams))
        ipln = np.asarray(ipln)
        ipar = np.asarray(ipar)
        ores = np.asarray(ores)
        oval = np.asarray(oval)
        opol = np.asarray(opol)

        vweights = np.ones(oval.size)
        if self.pol_grad > 0:
            pweights = (1.0 - ores / 2.0) - oval
        else:
            pweights = np.ones(oval.size)

        oval = np_utils.to_categorical(ores, 3)

        for i in range(len(self.model)):
            print "Fitting model",i
            self.model[i].fit(x = [ipln, ipar], y = [oval, opol],
                  batch_size=self.batch_size,
                  sample_weight=[vweights, pweights],
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
                chunk = (count + EPD_CHUNK_SIZE - 1) / EPD_CHUNK_SIZE
                end_t = time.time()
                print "Time", int(end_t - start_t), "sec"
                print "Training on chunk ", chunk , " ending at position ", count, " with lr ", args.lr
                myNet.train(examples)
                if chunk % myNet.rsavo == 0:
                    myNet.save_checkpoint("nets","ID-" + str(chunk), args, True)
                elif chunk % myNet.rsav == 0:
                    myNet.save_checkpoint("nets","ID-" + str(chunk), args, False)

                examples = []
                start_t = time.time()
                print "Collecting data" 

            #break out
            if not line:
                break

            # add to examples
            examples.append(line)

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epd','-e', dest='epd', required=False, help='Path to labeled EPD file for training')
    parser.add_argument('--trn','-t', dest='trn', required=False, help='Path to labeled training file')
    parser.add_argument('--id','-i', dest='id', required=False, type=int, default=0, help='ID of neural network to load.')
    parser.add_argument('--batch-size','-b',dest='batch_size', required=False, type=int, default=4096, help='Training batch size.')
    parser.add_argument('--epochs',dest='epochs', required=False, type=int, default=1, help='Training epochs.')
    parser.add_argument('--learning-rate','-l',dest='lr', required=False, type=float, default=0.001, help='Training learning rate.')
    parser.add_argument('--vald-split',dest='vald_split', required=False, type=float, default=0.0, help='Fraction of sample to use for validation.')
    parser.add_argument('--chunk-size',dest='chunk_size', required=False, type=int, default=4096, help='PGN chunk size.')
    parser.add_argument('--cores',dest='cores', required=False, type=int, default=multiprocessing.cpu_count(), help='Number of cores to use.')
    parser.add_argument('--gpus',dest='gpus', required=False, type=int, default=0, help='Number of gpus to use.')
    parser.add_argument('--gzip','-z',dest='gzip', required=False, action='store_true',help='Process zipped file.')
    parser.add_argument('--nets',dest='nets', nargs='+', required=False, type=int, default=[0,1,2], \
                        help='Nets to train from 0=2x32,6x64,12x128,20x256,4=40x256.')
    parser.add_argument('--rsav',dest='rsav', required=False, type=int, default=1, help='Save graph every RSAV chunks.')
    parser.add_argument('--rsavo',dest='rsavo', required=False, type=int, default=20, help='Save optimization state every RSAVO chunks.')
    parser.add_argument('--rand',dest='rand', required=False, action='store_true', help='Generate random network.')
    parser.add_argument('--opt',dest='opt', required=False, type=int, default=1, help='Optimizer 0=SGD 1=Adam.')
    parser.add_argument('--pol',dest='policy', required=False, type=int,default=1, help='Policy head style 0=Lc0 styel, 1=A0 style')
    parser.add_argument('--pol_w',dest='pol_w', required=False, type=float, default=1.0, help='Policy loss weight.')
    parser.add_argument('--val_w',dest='val_w', required=False, type=float, default=1.0, help='Value loss weight.')
    parser.add_argument('--pol_grad',dest='pol_grad', required=False, type=int, default=0, help='0=standard 1=multiply policy by score.')

    args = parser.parse_args()
    
    myNet = NNet(args)

    global USE_EPD
    global EPD_CHUNK_SIZE
    EPD_CHUNK_SIZE = args.chunk_size * 80
    chunk = args.id

    myNet.load_checkpoint("nets","ID-" + str(chunk), args)

    myNet.compile_model(args)

    if args.rand:
        myNet.save_checkpoint("nets","ID-" + str(chunk), args, False)
    elif (args.epd != None) or (args.trn != None):
        start = chunk * EPD_CHUNK_SIZE + 1
        if args.epd != None:
            USE_EPD = True
            train_epd(myNet, args, args.epd, args.gzip, start)
        else:
            USE_EPD = False
            train_epd(myNet, args, args.trn, args.gzip, start)


if __name__ == "__main__":
    main(sys.argv[1:])
