import sys
import os
import time
import chess
import chess.pgn
import resnet
import argparse
import time
import gzip
import numpy as np
import multiprocessing
from joblib import Parallel, delayed

from keras.models import load_model
from keras import optimizers
from keras.utils.training_utils import multi_gpu_model
from keras import backend as K
import tensorflow as tf

CHANNELS = 24
NPARMS = 5
PGN_CHUNK_SIZE = 4096
EPD_CHUNK_SIZE = PGN_CHUNK_SIZE * 80

def fill_piece(iplanes, ix, bb, b, fk):
    abb = 0
    squares = chess.SquareSet(bb)
    for sq in squares:
        abb = abb | b.attacks_mask(sq)
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        if(fk < 4):
            iplanes[r, 7-f, ix + 12] = 1.0
        else:
            iplanes[r,  f,  ix + 12] = 1.0
    squares = chess.SquareSet(abb)
    for sq in squares:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        if(fk < 4):
            iplanes[r, 7-f, ix] = 1.0
        else:
            iplanes[r,  f,  ix] = 1.0


def fill_planes(iplanes, iparams, b):
    pl = chess.WHITE
    npl = chess.BLACK
    wksq = b.king(pl)
    fwksq = chess.square_file(wksq)
    #white piece attacks
    bb = b.kings & b.occupied_co[pl]
    fill_piece(iplanes,0,bb,b,fwksq)
    bb = b.queens & b.occupied_co[pl]
    fill_piece(iplanes,1,bb,b,fwksq)
    bb = b.rooks & b.occupied_co[pl]
    fill_piece(iplanes,2,bb,b,fwksq)
    bb = b.bishops & b.occupied_co[pl]
    fill_piece(iplanes,3,bb,b,fwksq)
    bb = b.knights & b.occupied_co[pl]
    fill_piece(iplanes,4,bb,b,fwksq)
    bb = b.pawns & b.occupied_co[pl]
    fill_piece(iplanes,5,bb,b,fwksq)
    #black piece attacks
    bb = b.kings & b.occupied_co[npl]
    fill_piece(iplanes,6,bb,b,fwksq)
    bb = b.queens & b.occupied_co[npl]
    fill_piece(iplanes,7,bb,b,fwksq)
    bb = b.rooks & b.occupied_co[npl]
    fill_piece(iplanes,8,bb,b,fwksq)
    bb = b.bishops & b.occupied_co[npl]
    fill_piece(iplanes,9,bb,b,fwksq)
    bb = b.knights & b.occupied_co[npl]
    fill_piece(iplanes,10,bb,b,fwksq)
    bb = b.pawns & b.occupied_co[npl]
    fill_piece(iplanes,11,bb,b,fwksq)
    #piece counts
    v = chess.popcount(b.queens & b.occupied_co[pl]) - chess.popcount(b.queens & b.occupied_co[npl])
    iparams[0] = v
    v = chess.popcount(b.rooks & b.occupied_co[pl]) - chess.popcount(b.rooks & b.occupied_co[npl])
    iparams[1] = v
    v = chess.popcount(b.bishops & b.occupied_co[pl]) - chess.popcount(b.bishops & b.occupied_co[npl])
    iparams[2] = v
    v = chess.popcount(b.knights & b.occupied_co[pl]) - chess.popcount(b.knights & b.occupied_co[npl])
    iparams[3] = v
    v = chess.popcount(b.pawns & b.occupied_co[pl]) - chess.popcount(b.pawns & b.occupied_co[npl])
    iparams[4] = v

def fill_examples(examples):
    epds, oval = list(zip(*examples))
    oval = list(oval)

    exams = []
    bb = chess.Board()
    for i,p in enumerate(epds):
        bb.set_epd(p)
        if bb.turn == chess.BLACK:
            bb = bb.mirror()
            if oval[i][0] == 1:
                oval[i] = [0.0, 0.0, 1.0]
            elif oval[i][2] == 1:
                oval[i] = [1.0, 0.0, 0.0]
        iplane = np.zeros(shape=(8,8,CHANNELS),dtype=np.float32)
        iparam = np.zeros(shape=(NPARMS),dtype=np.float32)
        fill_planes(iplane,iparam,bb)
        exams.append([iplane,iparam])

    return exams, oval

def build_model(cid):
    if cid == 1:
        return resnet.build_net((8, 8, CHANNELS), (NPARMS,),  2,  32)
    elif cid == 2:
        return resnet.build_net((8, 8, CHANNELS), (NPARMS,),  6,  64)
    elif cid == 3:
        return resnet.build_net((8, 8, CHANNELS), (NPARMS,), 12, 128)
    elif cid == 4:
        return resnet.build_net((8, 8, CHANNELS), (NPARMS,), 20, 256)
    elif cid == 5:
        return resnet.build_net((8, 8, CHANNELS), (NPARMS,), 40, 256)
    else:
        print "Unsupported network id (Use 1 to 5)."
        sys.exit()

class NNet():
    def __init__(self,args):
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.lr = args.lr
        self.cores = args.cores
        self.nets = args.nets
        self.rsav = args.rsav
        self.rsavo = args.rsavo

    def new_model(self,cid,args):
        if args.gpus > 1:
            with tf.device('/cpu:0'):
                return build_model(cid)
        else:
            return build_model(cid)

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
            self.model[i].compile(loss='categorical_crossentropy',
                  optimizer=self.opt,
                  metrics=['accuracy'])

    def set_lr(self,args,count):
        NPOS = args.npos
        if NPOS == 0:
            return

        if count <= NPOS/8:
            args.lr = 0.2
        elif count <= NPOS/2:
            args.lr = 0.02
        elif count <= 3 * NPOS/4:
            args.lr = 0.002
        else:
            args.lr = 0.0002

        for i in range(len(self.cpu_model)):
            K.set_value(self.model[i].optimizer.lr, args.lr)

    def train(self,examples):
        print "Generating input planes using", self.cores, "cores"
        start_t = time.time()

        nsz = len(examples)
        nlen = nsz / self.cores
        
        res = Parallel(n_jobs=self.cores)(delayed(fill_examples)\
            ( examples[ (id*nlen) : (min(nsz,(id+1)*nlen)) ] ) for id in range(self.cores))
        exams = []
        oval = []
        for i in range(self.cores):
            exams = exams + res[i][0]
            oval = oval + res[i][1]
        oval = np.asarray(oval)

        end_t = time.time()
        print "Time", int(end_t - start_t), "sec"
        
        ipls, ipars = list(zip(*exams))
        ipls = np.asarray(ipls)
        ipars = np.asarray(ipars)

        for i in range(len(self.model)):
            print "Fitting model",i
            self.model[i].fit(x = [ipls, ipars], y = oval,
                  batch_size=self.batch_size,
                  validation_split=0.1,
                  epochs=self.epochs)

    def predict(self, epd):
        bb = chess.Board()
        bb.set_epd(epd)
        inv = False
        if bb.turn == chess.BLACK:
            bb = bb.mirror()
            inv = True
        iplanes = np.zeros(shape=(8,8,CHANNELS),dtype=np.float32)
        iparams = np.zeros(shape=(NPARMS),dtype=np.float32)
        fill_planes(iplanes,iparams,bb)
        inp1 = iplanes[np.newaxis, :, :, :]
        inp2 = iparams[np.newaxis, :]
        v = self.model[0].predict([inp1, inp2])
        r = (v[0] * 1.0 + v[1] * 0.5)
        if inv:
            r = 1 - r
        return r

    def save_checkpoint(self, folder, filename, args, iopt=False):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.mkdir(folder)
        for i,n in enumerate(args.nets):
            fname = filepath  + "-model-" + str(n-1)
            self.cpu_model[i].save(fname, include_optimizer=iopt)

    def load_checkpoint(self, folder, filename, args):
        filepath = os.path.join(folder, filename)
        self.cpu_model = []
        for n in args.nets:
            fname = filepath  + "-model-" + str(n-1)
            if not os.path.exists(fname):
                self.cpu_model.append( self.new_model(n,args) )
            else:
                self.cpu_model.append( load_model(fname) )

def convert_pgn_to_epd(myPgn,myEpd,zipped=0,start=1):
    
    with (open(myPgn) if not zipped else gzip.open(myPgn)) as file, open(myEpd,'w') as efile:
        count = 0

        start_t = time.time()
        print "Collecting data"
        
        while True:

            #read game
            count = count + 1
            if count < start:
                for of in chess.pgn.scan_offsets(file):
                    if count >= start:
                        file.seek(of)
                        game = chess.pgn.read_game(file)
                        count = count + 1
                        break;
                    count = count + 1
                continue
            else:
                game = chess.pgn.read_game(file)

            #train network
            if (not game) or (count % PGN_CHUNK_SIZE == 0):
                chunk = (count + PGN_CHUNK_SIZE - 1) / PGN_CHUNK_SIZE
                end_t = time.time()
                print "Time", int(end_t - start_t), "sec"

                start_t = time.time()
                print "Collecting data" 

            #break out
            if not game:
                break

            #parse result
            sresult = game.headers["Result"]

            #iterate through the moves and add quiescent positions
            b = game.board()
            cap_prom_check = False or b.is_check()
            for move in game.main_line():
                if b.is_capture(move) or move.promotion:
                    cap_prom_check = True
                    b.push(move)
                else:
                    pfen = b.fen()
                    b.push(move)
                    ischeck = b.is_check()

                    if not (cap_prom_check or ischeck):
                        efile.write(pfen + ' ' + sresult + '\n')
                        
                    if ischeck:
                        cap_prom_check = True
                    else:
                        cap_prom_check = False

def train_pgn(myNet,args,myPgn,zipped=0,start=1):

    with (open(myPgn) if not zipped else gzip.open(myPgn)) as file:
        count = 0

        examples = []
        start_t = time.time()
        print "Collecting data"
        
        while True:

            #read game
            count = count + 1
            if count < start:
                for of in chess.pgn.scan_offsets(file):
                    if count >= start:
                        file.seek(of)
                        game = chess.pgn.read_game(file)
                        count = count + 1
                        break;
                    count = count + 1
                continue
            else:
                game = chess.pgn.read_game(file)

            #train network
            if (not game) or (count % PGN_CHUNK_SIZE == 0):
                chunk = (count + PGN_CHUNK_SIZE - 1) / PGN_CHUNK_SIZE
                end_t = time.time()
                print "Time", int(end_t - start_t), "sec"
                print "Training on chunk ", chunk, " ending at game ", count, " positions ", len(examples)
                myNet.train(examples)
                if chunk % myNet.rsavo == 0:
                    myNet.save_checkpoint("nets","ID-" + str(chunk), args, True)
                elif chunk % myNet.rsav == 0:
                    myNet.save_checkpoint("nets","ID-" + str(chunk), args, False)

                examples = []
                start_t = time.time()
                print "Collecting data" 

            #break out
            if not game:
                break

            #parse result
            sresult = game.headers["Result"]
            if sresult == '1-0':
                result = [1.0, 0.0, 0.0]
            elif sresult == '0-1':
                result = [0.0, 0.0, 1.0]
            else:
                result = [0.0, 1.0, 0.0]

            #iterate through the moves and add quiescent positions
            b = game.board()
            cap_prom_check = False or b.is_check()
            for move in game.main_line():
                if b.is_capture(move) or move.promotion:
                    cap_prom_check = True
                    b.push(move)
                else:
                    pfen = b.fen()
                    b.push(move)
                    ischeck = b.is_check()

                    if not (cap_prom_check or ischeck):
                        examples.append([pfen,result])
                        
                    if ischeck:
                        cap_prom_check = True
                    else:
                        cap_prom_check = False

def train_epd(myNet,args,myEpd,zipped=0,start=1):
    
    with (open(myEpd) if not zipped else gzip.open(myEpd)) as file:
        count = 0

        examples = []
        start_t = time.time()
        print "Collecting data"
        myNet.set_lr(args,count)

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
                myNet.set_lr(args,count)

            #break out
            if not line:
                break

            #get epd
            words = line.strip().split()
            epd = ''
            for i in range(0, len(words) - 1):
                epd = epd + words[i] + ' '

            # parse result
            sresult = words[-1]
            if sresult == '1-0':
                result = [1.0, 0.0, 0.0]
            elif sresult == '0-1':
                result = [0.0, 0.0, 1.0]
            else:
                result = [0.0, 1.0, 0.0]

            # add to examples
            examples.append([epd,result])

def play(myNet):
    b = chess.Board()
    while not b.is_game_over():
        print(b)
        while True:
            mvstr = raw_input("Your move: ")
            mv = chess.Move.from_uci(mvstr)
            if b.is_legal(mv):
                b.push(mv)
                break
            else:
                print "Illegal move"

        print(b)
        moves = b.generate_legal_moves()

        bestm = 0
        beste = 0
        for move in moves:
            b.push(move)
            eval = 1 - myNet.predict(b.fen())
            b.pop()
            if eval > beste:
                bestm = move
                beste = eval
            print move, eval

        b.push(bestm)
        print "My move: ", bestm, "Score ", 100 * beste


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epd','-e', dest='epd', required=False, help='Path to labeled EPD file for training')
    parser.add_argument('--pgn','-p', dest='pgn', required=False, help='Path to PGN file for training.')
    parser.add_argument('--id','-i', dest='id', required=False, type=int, default=0, help='ID of neural network to load.')
    parser.add_argument('--batch-size','-b',dest='batch_size', required=False, type=int, default=4096, help='Training batch size.')
    parser.add_argument('--epochs',dest='epochs', required=False, type=int, default=1, help='Training epochs.')
    parser.add_argument('--learning-rate','-l',dest='lr', required=False, type=float, default=0.001, help='Training learning rate.')
    parser.add_argument('--chunk-size',dest='chunk_size', required=False, type=int, default=4096, help='PGN chunk size.')
    parser.add_argument('--cores',dest='cores', required=False, type=int, default=multiprocessing.cpu_count(), help='Number of cores to use.')
    parser.add_argument('--gpus',dest='gpus', required=False, type=int, default=0, help='Number of gpus to use.')
    parser.add_argument('--gzip','-z',dest='gzip', required=False, action='store_true',help='Process zipped file.')
    parser.add_argument('--nets',dest='nets', nargs='+', required=False, type=int, default=[1,2,3], \
                        help='Nets to train from 1=2x32,6x64,12x128,20x256,5=40x256.')
    parser.add_argument('--rsav',dest='rsav', required=False, type=int, default=1, help='Save graph every RSAV chunks.')
    parser.add_argument('--rsavo',dest='rsavo', required=False, type=int, default=20, help='Save optimization state every RSAVO chunks.')
    parser.add_argument('--rand',dest='rand', required=False, action='store_true', help='Generate random network.')
    parser.add_argument('--npos',dest='npos', required=False, type=int, default=0, help='Number of positions in the training set.')
    parser.add_argument('--opt',dest='opt', required=False, type=int, default=1, help='Optimizer 0=SGD 1=Adam.')

    args = parser.parse_args()
    
    if args.pgn != None and args.epd != None:
        convert_pgn_to_epd(args.pgn, args.epd, args.gzip)
    else :
        myNet = NNet(args)

        global PGN_CHUNK_SIZE
        global EPD_CHUNK_SIZE
        PGN_CHUNK_SIZE = args.chunk_size
        EPD_CHUNK_SIZE = PGN_CHUNK_SIZE * 80
        chunk = args.id

        myNet.load_checkpoint("nets","ID-" + str(chunk), args)

        myNet.compile_model(args)

        if args.rand:
            myNet.save_checkpoint("nets","ID-" + str(chunk), args, False)
        elif args.pgn != None:
            start = chunk * PGN_CHUNK_SIZE + 1
            train_pgn(myNet, args, args.pgn, args.gzip, start)
        elif args.epd != None:
            start = chunk * EPD_CHUNK_SIZE + 1
            train_epd(myNet, args, args.epd, args.gzip, start)
        else:
            play(myNet)

if __name__ == "__main__":
    main(sys.argv[1:])
