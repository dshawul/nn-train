# nn-train
Chess neural network training program. This program takes chess games or result-labelled
epd positions and trains a neural network for prediction of the outcome of a game from a positon. 
So this is only a `value network` -- AlphaZero also has `policy network` that represent move probabilities.

	usage: train.py [-h] [--epd EPD] [--id ID] [--batch-size BATCH_SIZE]
	                [--epochs EPOCHS] [--learning-rate LR]
	                [--vald-split VALD_SPLIT] [--chunk-size CHUNK_SIZE]
	                [--cores CORES] [--gpus GPUS] [--gzip]
	                [--nets NETS [NETS ...]] [--rsav RSAV] [--rsavo RSAVO]
	                [--rand] [--npos NPOS] [--opt OPT] [--pol POLICY]

	optional arguments:
	  -h, --help            show this help message and exit
	  --epd EPD, -e EPD     Path to labeled EPD file for training
	  --id ID, -i ID        ID of neural network to load.
	  --batch-size BATCH_SIZE, -b BATCH_SIZE
	                        Training batch size.
	  --epochs EPOCHS       Training epochs.
	  --learning-rate LR, -l LR
	                        Training learning rate.
	  --vald-split VALD_SPLIT
	                        Fraction of sample to use for validation.
	  --chunk-size CHUNK_SIZE
	                        PGN chunk size.
	  --cores CORES         Number of cores to use.
	  --gpus GPUS           Number of gpus to use.
	  --gzip, -z            Process zipped file.
	  --nets NETS [NETS ...]
	                        Nets to train from 0=2x32,6x64,12x128,20x256,4=40x256.
	  --rsav RSAV           Save graph every RSAV chunks.
	  --rsavo RSAVO         Save optimization state every RSAVO chunks.
	  --rand                Generate random network.
	  --npos NPOS           Number of positions in the training set.
	  --opt OPT             Optimizer 0=SGD 1=Adam.
	  --pol POLICY          Policy head style 0=Lc0 styel, 1=A0 style



To train 2x32 and 6x64 networks from a gzipped labelled epd with result and best moves using
32 cpu cores and 4 gpus
    
    python train.py --gzip --epd quiet.epd.gz --nets 0 1 --cores 32 --gpus 4

Then to convert your keras model to protobuf tensorflow format:
    
    ./convert.sh ID-1-model-0

You will get an `ID-1-model-0.pb` in the nets/ directory. Rename it to something recognizable such 
as `net-6x64.pb` and put it some place it can be used by Scorpio.

To restart interrupted training from specific ID e.g. 120
    
    python train.py --epd quiet.epd --id 120

You can build your own network (different number of blocks and filters) by modifying resnet.py.

It should also be easy to use other types of networks -- as long as you keep the input planes the same.
The input planes are hard-wired in the probing code (egbbdll) so you won't be able to use your networks
unless you change that as well. In the future, I plan to offload that to the network file to give the user
maximum flexibility in designing new networks.

## Self play training

To train networks by reinforcement learning issue command
   
    ./selfplay.sh 3 2 1 0

This will train networks (20x256, 12x128, 6x64 and 2x32) using selfplay games produced
by the 20x256 network. The net used for producing selfplay games is mentioned first