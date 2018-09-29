# nn-train
Chess neural network training program. This program takes chess games or result-labelled
epd positions and trains a neural network for prediction of the outcome of a game from a positon. 
So this is only a `value network` -- AlphaZero also has `policy network` that represent move probabilities.

	usage: train.py [-h] [--epd EPD] [--pgn PGN] [--id ID]
	                [--batch-size BATCH_SIZE] [--epochs EPOCHS]
	                [--learning-rate LR] [--chunk-size CHUNK_SIZE] [--cores CORES]
	                [--gpus GPUS] [--gzip] [--nets NETS] [--rsav RSAV]
	                [--rsavo RSAVO]

	optional arguments:
	  -h, --help            show this help message and exit
	  --epd EPD, -e EPD     Path to labeled EPD file for training
	  --pgn PGN, -p PGN     Path to PGN file for training.
	  --id ID, -i ID        ID of neural network to load.
	  --batch-size BATCH_SIZE, -b BATCH_SIZE
	                        Training batch size.
	  --epochs EPOCHS       Training epochs.
	  --learning-rate LR, -l LR
	                        Training learning rate.
	  --chunk-size CHUNK_SIZE
	                        PGN chunk size.
	  --cores CORES         Number of cores to use.
	  --gpus GPUS           Number of gpus to use.
	  --gzip, -z            Process zipped file.
	  --nets NETS, -n NETS  Number of nets to train from
	                        2x32,6x64,12x128,20x256,40x256.
	  --rsav RSAV           Save graph every RSAV chunks.
	  --rsavo RSAVO         Save optimization state every RSAVO chunks.


To train from a collection of games (PGN), e.g. ccrl.pgn:
    
    python train.py --pgn ccrl.pgn

And to train an 2x32 net from gzip'ed file using 4 GPUs and 32 cores

    python train.py --gzip --pgn ccrl.pgn.gz --gpus 4 --cores 32 --nets 1

To train from a collection of result labelled (EPD) positions, e.g. quiet.epd:
    
    python train.py --epd quiet.epd

Training from epd files is faster because one doesn't have to parse PGN on the CPU.
Need to add input pipelining sometime in the future.

Then to convert your keras model to protobuf tensorflow format:
    
    ./convert.sh ID-1-model-0

To restart interrupted training from specific ID e.g. 120
    
    python train.py --epd quiet.epd --id 120

You will get an `ID-1-model-0.pb` in the nets/ directory. Rename it to something recognizable such 
as `net-6x64.pb` and put it some place it can be used by Scorpio.

You can build your own network (different number of blocks and filters) by modifying resnet.py.

It should also be easy to use other types of networks -- as long as you keep the input planes the same.
The input planes are hard-wired in the probing code (egbbdll) so you won't be able to use your networks
unless you change that as well. In the future, I plan to offload that to the network file to give the user
maximum flexibility in designing new networks.