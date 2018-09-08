# nn-train
Chess neural network training program. This program takes chess games or labelled
positions and trains a neural network to predict the outcome. So this is only a 
value network -- AlphaZero also has policy network that represent move probabilities.

To train from a collection of games (PGN) do:
    
    python train.py --pgn ccrl.pgn

And to train from a collection of labelled (EPD) positions:
    
    python train.ph --epd quiet.epd

Then to convert your keras model to protobuf tensorformat do:
    
    ./convert.sh ID-model-0

You will get ID-model-0.pb in the nets/ directory. Rename it to
something recognizable as net-6x64.pb and put it some place it can be used.

You can use multiple GPUs for training, you can also train multiple networks 
simultanesouly by editing code in the NNet() class in train.py.
Eg. To train a 2x32 and 6x64 networks simultaneously do:

	self.model.append( resnet.ResnetBuilder.build_resnet_2x32((8, 8, CHANNELS), (NPARMS,)) )
	self.model.append( resnet.ResnetBuilder.build_resnet_6x64((8, 8, CHANNELS), (NPARMS,)) )
	# self.model.append( resnet.ResnetBuilder.build_resnet_12x64((8, 8, CHANNELS), (NPARMS,)) )
	# self.model.append( resnet.ResnetBuilder.build_resnet_12x128((8, 8, CHANNELS), (NPARMS,)) )
	# self.model.append( resnet.ResnetBuilder.build_resnet_24x128((8, 8, CHANNELS), (NPARMS,)) )
	# self.model.append( resnet.ResnetBuilder.build_resnet_20x256((8, 8, CHANNELS), (NPARMS,)) )
	# self.model.append( resnet.ResnetBuilder.build_resnet_40x256((8, 8, CHANNELS), (NPARMS,)) )

You can also build your own network (different number of blocks and filters) by modififying resnet.py.

It should be easy to use other types of networks as well.