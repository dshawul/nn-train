# nn-train
Chess neural network training program. This program takes chess games or result-labelled
epd positions and trains a neural network for prediction of the outcome of a game from a positon. 
So this is only a `value network` -- AlphaZero also has `policy network` that represent move probabilities.

To train from a collection of games (PGN), e.g. ccrl.pgn:
    
    python train.py --pgn ccrl.pgn

To train from a collection of result labelled (EPD) positions, e.g. quiet.epd:
    
    python train.py --epd quiet.epd

Then to convert your keras model to protobuf tensorflow format:
    
    ./convert.sh ID-model-0

You will get an `ID-model-0.pb` in the nets/ directory. Rename it to something recognizable such 
as `net-6x64.pb` and put it some place it can be used by Scorpio.

You can use multiple GPUs for training. You can also train multiple networks 
simultanesouly by editing relevant code, NNet() class contructor in train.py.
Eg. To train a 2x32 and 6x64 networks simultaneously do:

	self.model.append( resnet.ResnetBuilder.build_resnet_2x32((8, 8, CHANNELS), (NPARMS,)) )
	self.model.append( resnet.ResnetBuilder.build_resnet_6x64((8, 8, CHANNELS), (NPARMS,)) )
	# self.model.append( resnet.ResnetBuilder.build_resnet_12x128((8, 8, CHANNELS), (NPARMS,)) )
	# self.model.append( resnet.ResnetBuilder.build_resnet_20x256((8, 8, CHANNELS), (NPARMS,)) )
	# self.model.append( resnet.ResnetBuilder.build_resnet_40x256((8, 8, CHANNELS), (NPARMS,)) )

You can also build your own network (different number of blocks and filters) by modifying resnet.py.

It should also be easy to use other types of networks -- as long as you keep the input planes the same.
The input planes are hard-wired in the probing code (egbbdll) so you won't be able to use your networks
unless you change that as well. In the future, I plan to offload that to the network file to give the user
maximum flexibility in designing new networks.