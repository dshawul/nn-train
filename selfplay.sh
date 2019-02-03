#!/bin/bash

set -e

#setup parameters for selfplay
SC=~/Scorpio # path to scorpio exec
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
    python train.py --rand --nets $(($1+1))
    ./convert.sh ID-0-model-$1
    if [ $GPUS -gt 0 ]; then
        convert-to-uff nets/ID-0-model-$1.pb
        cp nets/ID-0-model-$1.uff nets/hist/ID-0-model-$1.uff
    fi
    cp nets/ID-0-model-$1 nets/hist/ID-0-model-$1
    cp nets/ID-0-model-$1.pb nets/hist/ID-0-model-$1.pb
    rm -rf ${SC}/allgames.pgn ${SC}/games.pgn
}

fornets() {
    for i in "${net[@]}"; do
        if ! [ -z $i ]; then
            $1 $i
        fi
    done
}

if [ -e nets ]; then
    if [ "$1" = "init" ]; then
        read -p "Are you sure you want to re-initialize [y/N]:" A
        A=${A:-n}
        case "$A" in
            [nN]) 
                ;;
            *)
                init0
                if ! [ -z "$2" ]; then net[0]=$2; fi
                if ! [ -z "$3" ]; then net[1]=$3; fi
                if ! [ -z "$4" ]; then net[2]=$4; fi
                if ! [ -z "$5" ]; then net[3]=$5; fi
                if ! [ -z "$6" ]; then net[4]=$6; fi
                fornets init
                ;;
        esac
    fi
else
    init0
    if ! [ -z "$1" ]; then net[0]=$1; fi
    if ! [ -z "$2" ]; then net[1]=$2; fi
    if ! [ -z "$3" ]; then net[2]=$3; fi
    if ! [ -z "$4" ]; then net[3]=$4; fi
    if ! [ -z "$5" ]; then net[4]=$5; fi
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
V=`ls -l nets/hist/*-model-${Pnet}.pb | grep net | wc -l`
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
    mv nets/ID-1-model-$1 nets/ID-0-model-$1
    ./convert.sh ID-0-model-$1
    if [ $GPUS -gt 0 ]; then
        convert-to-uff nets/ID-0-model-$1.pb
    fi
}

#driver loop
while true ; do

    fornets move

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

    fornets train

    fornets conv

    V=$((V+1))
done
