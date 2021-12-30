# nn-train
Chess (and variants) neural network (NN) and Efficiently Updatable Neural Network (NNUE) training program.
This program takes labelled epd positions and trains a neural network to predict the outcome and/or move choice
for a position.

      usage: train.py [-h] [--epd EPD] [--dir DIR] [--id ID]
                      [--global-steps GLOBAL_STEPS] [--batch-size BATCH_SIZE]
                      [--learning-rate LR] [--validation-split VALD_SPLIT]
                      [--cores CORES] [--gpus GPUS] [--gzip] [--net NET]
                      [--rsav RSAV] [--rsavo RSAVO] [--rand] [--opt OPT]
                      [--policy-channels POL_CHANNELS] [--policy-weight POL_W]
                      [--value-weight VAL_W] [--score-weight SCORE_W]
                      [--policy-gradient POL_GRAD] [--no-auxinp]
                      [--channels CHANNELS] [--boardx BOARDX] [--boardy BOARDY]
                      [--frac-z FRAC_Z] [--frac-pi FRAC_PI] [--piece-map PCMAP]
                      [--mixed] [--head-type HEAD_TYPE] [--max-steps MAX_STEPS]

      optional arguments:
        -h, --help            show this help message and exit
        --epd EPD, -e EPD     Path to labeled EPD file for training
        --dir DIR             Path to network files
        --id ID, -i ID        ID of neural networks to load.
        --global-steps GLOBAL_STEPS
                              Global number of steps trained so far.
        --batch-size BATCH_SIZE, -b BATCH_SIZE
                              Training batch size.
        --learning-rate LR, -l LR
                              Training learning rate.
        --validation-split VALD_SPLIT
                              Fraction of sample to use for validation.
        --cores CORES         Number of cores to use.
        --gpus GPUS           Number of gpus to use.
        --gzip, -z            Process zipped file.
        --net NET             Net to train from
                              0=2x32,6x64,12x128,20x256,4=30x384,5=NNUE.
        --rsav RSAV           Save graph every RSAV steps.
        --rsavo RSAVO         Save optimization state every RSAVO steps.
        --rand                Generate random network.
        --opt OPT             Optimizer 0=SGD 1=Adam.
        --policy-channels POL_CHANNELS
                              Number of policy channels
        --policy-weight POL_W
                              Policy loss weight.
        --value-weight VAL_W  Value loss weight.
        --score-weight SCORE_W
                              Score loss weight.
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
        --head-type HEAD_TYPE
                              Heads of neural network, 0=value/policy,
                              1=value/score, 2=all three, 3=value only.
        --max-steps MAX_STEPS
                              Maximum number of steps to train for.

To train 2x32 networks from a gzipped labelled epd with result and best moves using
16 cpu cores and 1 gpu
    
    python src/train.py --dir nets --gzip --epd quiet.epd.gz --net 0 --cores 16 --gpus 1

Then to convert your keras model to protobuf tensorflow format:
    
    ./scripts/convert-to-pb.sh nets/ID-1-model-0

To also convert to UFF format use

    ./scripts/prepare.sh nets 1 0

To restart interrupted training from specific ID e.g. 120
    
    python src/train.py --epd quiet.epd --id 120

You can build your own network (different number of blocks and filters) by modifying resnet.py.

## Self play training

To train networks by reinforcement learning issue command
   
    ./train-selfplay.sh 3

This will train networks 20x256 resnet using selfplay games produced
by the 20x256 network. The net used for producing selfplay games is mentioned first
