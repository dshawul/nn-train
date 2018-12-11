#!/bin/bash

set -e

#setup parameters for selfplay
SC=../Scorpio # path to scorpio exec
SV=800        # mcts simulations
G=2500        # games per worker
OPT=1         # Optimizer 0=SGD 1=ADAM
LR=0.001      # learning rate
EPOCHS=1      # Number of epochs

#display help
display_help() {
    echo "Usage: $0 [Option...] {ID} [init]" >&2
    echo
    echo "   -h     Display this help message."
    echo "   ID     Network ID to train 0..4 for 2x32,6x64,12x128,20x256,40x356."
    echo "   init   Reinitialize network training from scratch (optional)"
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
    NDIR=$PWD/nets/ID-0-model-$1.uff
else
    GPUS=0
    NDIR=$PWD/nets/ID-0-model-$1.pb
fi

#initialize network
init() {
    echo "Initializing training."
    rm -rf nets
    mkdir nets 
    mkdir nets/hist
    python train.py --rand --nets $(($1+1))
    ./convert.sh ID-0-model-$1
    if [ $GPUS -gt 0 ]; then
        convert-to-uff nets/ID-0-model-$1.pb
        cp nets/ID-0-model-$1.uff nets/hist/net-0.uff
    fi
    cp nets/ID-0-model-$1 nets/hist/net-0
    cp nets/ID-0-model-$1.pb nets/hist/net-0.pb
    rm -rf ${SC}/allgames.pgn ${SC}/games.pgn
}

if [ -e nets ]; then
    if [ "$2" = "init" ]; then
        read -p "Are you sure you want to re-initialize [y/N]:" A
        A=${A:-n}
        case "$A" in
            [nN]) 
                ;;
            *)
                init $1;;
        esac
    fi
else
    init $1
fi

#start network id
V=`ls -l nets/hist/*.pb | grep net | wc -l`
V=$((V-1))

#run selfplay
run() {
    export CUDA_VISIBLE_DEVICES="$1" 
    taskset $3 time ./scorpio nn_path ${NDIR} new book off sv ${SV} pvstyle 1 selfplay $2 games$1.pgn quit
}

#use all gpus
rungames() {
    if [ $GPUS = 0 ]; then
        M=$((2**CPUS-1))
        run 0 $1 $M &
    else
        I=$((CPUS/GPUS))
        M=$((2**I-1))
        for k in `seq 0 $((GPUS-1))`; do
            S=$((k*I))
            run $k $1 $((M<<S)) &
        done
    fi
    wait
    echo "All jobs finished"
}

#train network
train() {
    python train.py --epd ${SC}/games.epd --nets $(($1+1)) --gpus ${GPUS} --cores ${CPUS} --opt ${OPT} --learning-rate ${LR} --epochs ${EPOCHS}
}

#driver loop
while true ; do
    cp nets/ID-0-model-$1 nets/hist/net-${V}
    cp nets/ID-0-model-$1.pb nets/hist/net-${V}.pb
    if [ $GPUS -gt 0 ]; then
        cp nets/ID-0-model-$1.uff nets/hist/net-${V}.uff
    fi

    cd ${SC}
    
    rungames ${G}
    rm -rf games.pgn
    cat games*.pgn > games.pgn
    cat games.pgn >> allgames.pgn
    ./scorpio pgn_to_epd games.pgn games.epd quit
    rm -rf games.pgn games*.pgn
    shuf games.epd >x
    mv x games.epd

    cd -

    train $1
    mv nets/ID-1-model-$1 nets/ID-0-model-$1
    ./convert.sh ID-0-model-$1
    if [ $GPUS -gt 0 ]; then
        convert-to-uff nets/ID-0-model-$1.pb
    fi
    V=$((V+1))
done
