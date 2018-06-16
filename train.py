import sys
import os
import time
import chess
import chess.pgn
import resnet
import numpy as np

from keras.models import load_model
from keras import optimizers

############# Workaround for multi-gpu training and model saving bug #################
from keras.layers import Lambda, concatenate
from keras import Model
import tensorflow as tf

def multi_gpu_model(model, gpus):
  if isinstance(gpus, (list, tuple)):
    num_gpus = len(gpus)
    target_gpu_ids = gpus
  else:
    num_gpus = gpus
    target_gpu_ids = range(num_gpus)

  def get_slice(data, i, parts):
    shape = tf.shape(data)
    batch_size = shape[:1]
    input_shape = shape[1:]
    step = batch_size // parts
    if i == num_gpus - 1:
      size = batch_size - step * i
    else:
      size = step
    size = tf.concat([size, input_shape], axis=0)
    stride = tf.concat([step, input_shape * 0], axis=0)
    start = stride * i
    return tf.slice(data, start, size)

  all_outputs = []
  for i in range(len(model.outputs)):
    all_outputs.append([])

  # Place a copy of the model on each GPU,
  # each getting a slice of the inputs.
  for i, gpu_id in enumerate(target_gpu_ids):
    with tf.device('/gpu:%d' % gpu_id):
      with tf.name_scope('replica_%d' % gpu_id):
        inputs = []
        # Retrieve a slice of the input.
        for x in model.inputs:
          input_shape = tuple(x.get_shape().as_list())[1:]
          slice_i = Lambda(get_slice,
                           output_shape=input_shape,
                           arguments={'i': i,
                                      'parts': num_gpus})(x)
          inputs.append(slice_i)

        # Apply model on slice
        # (creating a model replica on the target device).
        outputs = model(inputs)
        if not isinstance(outputs, list):
          outputs = [outputs]

        # Save the outputs for merging back together later.
        for o in range(len(outputs)):
          all_outputs[o].append(outputs[o])

  # Merge outputs on CPU.
  with tf.device('/cpu:0'):
    merged = []
    for name, outputs in zip(model.output_names, all_outputs):
      merged.append(concatenate(outputs,
                                axis=0, name=name))
    return Model(model.inputs, merged)
#####################################

CHANNELS = 12 #18
NPARMS = 5
PGN_CHUNK_SIZE = 4096
EPD_CHUNK_SIZE = PGN_CHUNK_SIZE * 80

def fill_piece(iplanes, ix, bb, b, fk):
    abb = 0
    squares = chess.SquareSet(bb)
    for sq in squares:
        abb = abb | b.attacks_mask(sq)
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

class NNet():
    def __init__(self):
        self.batch_size = 4096
        self.epochs = 20
        self.lr = 0.001
        self.model = []
        self.model.append( resnet.ResnetBuilder.build_resnet_1((8, 8, CHANNELS), (NPARMS,)) )
        self.model.append( resnet.ResnetBuilder.build_resnet_3((8, 8, CHANNELS), (NPARMS,)) )
        self.model.append( resnet.ResnetBuilder.build_resnet_6((8, 8, CHANNELS), (NPARMS,)) )
        # self.model.append( resnet.ResnetBuilder.build_resnet_12((8, 8, CHANNELS), (NPARMS,)) )
        # self.model.append( resnet.ResnetBuilder.build_resnet_20((8, 8, CHANNELS), (NPARMS,)) )
        # self.model.append( resnet.ResnetBuilder.build_resnet_40((8, 8, CHANNELS), (NPARMS,)) )
        self.opt = optimizers.Adam(lr=self.lr)
        for i in range(len(self.model)):
            self.model[i] = multi_gpu_model(self.model[i], gpus=4)
            self.model[i].compile(loss='mean_squared_error',
                  optimizer=self.opt,
                  metrics=['accuracy'])
        self.b = chess.Board()
    
    def train(self,examples):
        epds, oval = list(zip(*examples))
        oval = list(oval)
        oval = np.asarray(oval)

        exams = []
        for i,p in enumerate(epds):
            self.b.set_epd(p)
            if self.b.turn == chess.BLACK:
                self.b = self.b.mirror()
                oval[i] = 1 - oval[i]
            iplane = np.zeros(shape=(8,8,CHANNELS),dtype=np.float32)
            iparam = np.zeros(shape=(NPARMS),dtype=np.float32)
            fill_planes(iplane,iparam,self.b)
            exams.append([iplane,iparam])
                
        ipls, ipars = list(zip(*exams))
        ipls = np.asarray(ipls)
        ipars = np.asarray(ipars)

        for i in range(len(self.model)):
            print "Fitting model",i
            self.model[i].fit(x = [ipls, ipars], y = oval,
                  batch_size=self.batch_size,
                  epochs=self.epochs)

    def predict(self, epd):
        self.b.set_epd(epd)
        inv = False
        if self.b.turn == chess.BLACK:
            self.b = self.b.mirror()
            inv = True
        iplanes = np.zeros(shape=(8,8,CHANNELS),dtype=np.float32)
        iparams = np.zeros(shape=(NPARMS),dtype=np.float32)
        fill_planes(iplanes,iparams,self.b)
        inp1 = iplanes[np.newaxis, :, :, :]
        inp2 = iparams[np.newaxis, :]
        v = self.model[0].predict([inp1, inp2])
        if inv:
            return 1 - v[0]
        else:
            return v[0]

    def save_checkpoint(self, folder, filename):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.mkdir(folder)
        for i,m in enumerate(self.model):
            m.save(filepath + "-model-" + str(i))

    def load_checkpoint(self, folder, filename):
        filepath = os.path.join(folder, filename)
        self.model[0] = load_model(filepath  + "-model-" + str(0), {'tf': tf})


def train_pgn(myNet,myPgn,start=1):

    with open(myPgn) as file:
        examples = []
        count = 0
        print "Collecting data"
        
        while True:

            #read game
            count = count + 1
            if count < start:
                game = chess.pgn.scan_headers(file)
                continue
            else:
                game = chess.pgn.read_game(file)

            #train network
            if (not game) or (count % PGN_CHUNK_SIZE == 0):
                chunk = (count + PGN_CHUNK_SIZE - 1) / PGN_CHUNK_SIZE
                print "Training on chunk ", chunk, " ending at game ", count, " positions ", len(examples)
                myNet.train(examples)
                myNet.save_checkpoint("nets","ID-" + str(chunk))
                examples = []
                print "Collecting data" 

            #break out
            if not game:
                break

            #parse result
            sresult = game.headers["Result"]
            if sresult == '1-0':
                result = 1.0
            elif sresult == '0-1':
                result = 0.0
            else:
                result = 0.5

            #iterate through the moves
            b = game.board()
            examples.append([b.fen(),result])
            for move in game.main_line():
                b.push(move)
                examples.append([b.fen(),result])


def train_epd(myNet,myEpd,start=1):

    with open(myEpd) as file:
        examples = []
        count = 0
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
                print "Training on chunk ", chunk , " ending at position ", count
                myNet.train(examples)
                myNet.save_checkpoint("nets","ID-" + str(chunk))
                examples = []
                print "Collecting data" 

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
                result = 1.0
            elif sresult == '0-1':
                result = 0.0
            else:
                result = 0.5

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
    myNet = NNet()
    # EPD_PATH = "../tune/merged.epd"
    PGN_PATH = "../../ccrl.pgn"

    #start training from specific chunk
    chunk = 0
    start = 1
    if chunk > 0:
        # start = chunk * EPD_CHUNK_SIZE + 1
        start = chunk * PGN_CHUNK_SIZE + 1
        myNet.load_checkpoint("nets","ID-" + str(chunk))

    # train_epd(myNet, EPD_PATH, start)
    train_pgn(myNet, PGN_PATH, start)

    # play(myNet)

if __name__ == "__main__":
    main(sys.argv[1:])
