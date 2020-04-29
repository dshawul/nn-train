# nn-train
Chess (and variants) neural network training program. This program takes result-labelled epd positions and 
trains a neural network to predict the outcome and move choice for a position.

      usage: train.py [-h] [--epd EPD] [--dir DIR] [--id ID]
                      [--batch-size BATCH_SIZE] [--nbatch NBATCH] [--epochs EPOCHS]
                      [--learning-rate LR] [--validation-split VALD_SPLIT]
                      [--cores CORES] [--gpus GPUS] [--gzip]
                      [--nets NETS [NETS ...]] [--rsav RSAV] [--rsavo RSAVO]
                      [--rand] [--opt OPT] [--policy-channels POL_CHANNELS]
                      [--policy-weight POL_W] [--value-weight VAL_W]
                      [--policy-gradient POL_GRAD] [--no-auxinp]
                      [--channels CHANNELS] [--boardx BOARDX] [--boardy BOARDY]
                      [--frac-z FRAC_Z] [--frac-pi FRAC_PI] [--piece-map PCMAP]
                      [--mixed]

      optional arguments:
        -h, --help            show this help message and exit
        --epd EPD, -e EPD     Path to labeled EPD file for training
        --dir DIR             Path to network files
        --id ID, -i ID        ID of neural network to load.
        --batch-size BATCH_SIZE, -b BATCH_SIZE
                              Training batch size.
        --nbatch NBATCH       Number of batches to process at one time.
        --epochs EPOCHS       Training epochs.
        --learning-rate LR, -l LR
                              Training learning rate.
        --validation-split VALD_SPLIT
                              Fraction of sample to use for validation.
        --cores CORES         Number of cores to use.
        --gpus GPUS           Number of gpus to use.
        --gzip, -z            Process zipped file.
        --nets NETS [NETS ...]
                              Nets to train from
                              0=2x32,6x64,12x128,20x256,4=24x320,5=30x384,6=40x512.
        --rsav RSAV           Save graph every RSAV chunks.
        --rsavo RSAVO         Save optimization state every RSAVO chunks.
        --rand                Generate random network.
        --opt OPT             Optimizer 0=SGD 1=Adam.
        --policy-channels POL_CHANNELS
                              Number of policy channels
        --policy-weight POL_W
                              Policy loss weight.
        --value-weight VAL_W  Value loss weight.
        --policy-gradient POL_GRAD
                              0=standard 1=multiply policy by score.
        --no-auxinp, -u       Don't use auxillary input
        --channels CHANNELS, -c CHANNELS
                              number of input channels of network.
        --boardx BOARDX, -x BOARDX
                              board x-dimension.
        --boardy BOARDY, -y BOARDY
                              board y-dimension.
        --frac-z FRAC_Z       Fraction of ouctome(Z) relative to MCTS value(Q).
        --frac-pi FRAC_PI     Fraction of MCTS policy (PI) relative to one-hot
                              policy(P).
        --piece-map PCMAP     Map pieces to planes
        --mixed               Use mixed precision training


To train 2x32 and 6x64 networks from a gzipped labelled epd with result and best moves using
32 cpu cores and 4 gpus
    
    python src/train.py --dir nets --gzip --epd quiet.epd.gz --nets 0 1 --cores 32 --gpus 4

Then to convert your keras model to protobuf tensorflow format:
    
    ./scripts/convert-to-pb.sh nets/ID-1-model-0

To also convert to UFF format use

    ./scripts/prepare.sh nets 1

To restart interrupted training from specific ID e.g. 120
    
    python src/train.py --epd quiet.epd --id 120

You can build your own network (different number of blocks and filters) by modifying resnet.py.

## Self play training

To train networks by reinforcement learning issue command
   
    ./selfplay.sh 3 2 1 0

This will train networks (20x256, 12x128, 6x64 and 2x32) using selfplay games produced
by the 20x256 network. The net used for producing selfplay games is mentioned first
