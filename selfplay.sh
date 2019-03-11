#!/bin/bash

set -e

#setup parameters for selfplay
SC=../Scorpio  # path to scorpio exec
SV=800         # mcts simulations
G=8064         # games per worker
OPT=1          # Optimizer 0=SGD 1=ADAM
LR=0.001       # learning rate
EPOCHS=1       # Number of epochs
NREPLAY=500000 # Number of games in the replay buffer
NSTEPS=250     # Number of steps
CPUCT=150      # Cpuct constant
POL_TEMP=100   # Policy temeprature
NOISE_FRAC=25  # Fraction of Dirchilet noise

#display help
display_help() {
    echo "Usage: $0 [Option...] {IDs} " >&2
    echo
    echo "   -h     Display this help message."
    echo "   IDs    Network IDs to train 0..4 for 2x32,6x64,12x128,20x256,40x356."
    echo
}

if [ "$1" == "-h" ]; then
  display_help
  exit 0
fi

#check if Scorpio directory exists
if [ ! -f ${SC}/scorpio ]; then
    echo "Please set the correct path to scorpio."
    exit 0
fi

#number of cpus and gpus
CPUS=`grep -c ^processor /proc/cpuinfo`
if [ ! -z `which nvidia-smi` ]; then
    GPUS=`nvidia-smi | grep "N/A" | wc -l`
else
    GPUS=0
fi

#initialize network
init0() {
    echo "Initializing training."
    rm -rf nets
    mkdir nets 
    mkdir nets/hist
}

init() {
    python train.py --rand --nets $1
    ./convert.sh ID-0-model-$1
    if [ $GPUS -gt 0 ]; then
        convert-to-uff nets/ID-0-model-$1.pb -O value/Softmax -O policy/Reshape
        cp nets/ID-0-model-$1.uff nets/hist/ID-0-model-$1.uff
    fi
    cp nets/ID-0-model-$1 nets/hist/ID-0-model-$1
    cp nets/ID-0-model-$1.pb nets/hist/ID-0-model-$1.pb
}

fornets() {
    for i in "${net[@]}"; do
        if ! [ -z $i ]; then
            $1 $i
        fi
    done
}

if ! [ -z "$1" ]; then net[0]=$1; fi
if ! [ -z "$2" ]; then net[1]=$2; fi
if ! [ -z "$3" ]; then net[2]=$3; fi
if ! [ -z "$4" ]; then net[3]=$4; fi
if ! [ -z "$5" ]; then net[4]=$5; fi

if ! [ -e nets ]; then
    init0
    fornets init
fi

#which net format to use
Pnet=${net[0]}
if [ $GPUS -gt 0 ]; then
    NDIR=$PWD/nets/ID-0-model-${Pnet}.uff
else
    NDIR=$PWD/nets/ID-0-model-${Pnet}.pb
fi

#start network id
V=`find nets/hist/ID-*-model-${Pnet}.pb -type f | grep -o ID-[0-9]* | grep -o [0-9]* | sort -rn | head -1`

#run selfplay
run() {
    export CUDA_VISIBLE_DEVICES="$1" 
    SCOPT="reuse_tree 0 fpu_is_loss 0 fpu_red 0 cpuct_init ${CPUCT} policy_temp ${POL_TEMP} noise_frac ${NOISE_FRAC}"
    taskset -c $3 time ./scorpio nn_path ${NDIR} new ${SCOPT} sv ${SV} pvstyle 1 selfplayp $2 games$1.pgn quit
}

#use all gpus
rungames() {
    if [ $GPUS = 0 ]; then
        run 0 $1 0-$CPUS:1 &
    else
        I=$((CPUS/GPUS))
        for k in `seq 0 $((GPUS-1))`; do
            run $k $1 $((k*I))-$((k*I+I-1)):1 &
        done
    fi
    wait
    echo "All jobs finished"
}

#train network
train() {
    python train.py --epd nets/temp.epd --nets ${net[@]} --gpus ${GPUS} --cores ${CPUS} --opt ${OPT} --learning-rate ${LR} --epochs ${EPOCHS}
}

#move
move() {
    cp nets/ID-0-model-$1 nets/hist/ID-${V}-model-$1
    cp nets/ID-0-model-$1.pb nets/hist/ID-${V}-model-$1.pb
    if [ $GPUS -gt 0 ]; then
        cp nets/ID-0-model-$1.uff nets/hist/ID-${V}-model-$1.uff
    fi
}

#convert
conv() {
    E=`ls -l nets/ID-*-model-$1 | wc -l`
    E=$((E-1))
    mv nets/ID-$E-model-$1 nets/ID-0-model-$1
    rm -rf nets/ID-[1-$E]-model-$1
    ./convert.sh ID-0-model-$1
    if [ $GPUS -gt 0 ]; then
        convert-to-uff nets/ID-0-model-$1.pb -O value/Softmax -O policy/Reshape
        rm -rf nets/*.trt
    fi
}

#get selfplay games
get_selfplay_games() {
    cd ${SC}
    rungames ${G}
    rm -rf games.pgn
    cat games*.pgn > cgames.pgn
    rm -rf games*.pgn
    cd -
    mv ${SC}/cgames.pgn .
}

#prepare training data
prepare() {

    rm -rf cgames.pgn

    #run games
    get_selfplay_games

    #convert to epd
    cat cgames.pgn >> nets/allgames.pgn
    ${SC}/scorpio pgn_to_epd cgames.pgn nets/data$V.epd quit

    #prepare shuffled replay buffer
    if [ $GPUS -gt 0 ]; then
        ND=$((NREPLAY/(GPUS*G)))
    else
        ND=$((NREPLAY/(CPUS*G)))
    fi
    if [ $ND -ge $V ]; then
        ND=$V
    else
        A=`seq 0 $((V-ND-1))`
        for i in $A; do
            rm -rf nets/data$i.epd
        done
    fi
    cat nets/data*.epd > nets/temp.epd

    shuf -n $((NSTEPS * 4096)) nets/temp.epd >x
    mv x nets/temp.epd
}

#Selfplay training loop
selfplay_loop() {
    while true ; do
        fornets move

        prepare

        train

        fornets conv

        V=$((V+1))
    done
}

selfplay_loop
