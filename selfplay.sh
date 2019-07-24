#!/bin/bash

set -e

#set working directory and executable
SC=nn-dist/Scorpio-train/bin/Linux
EXE=scorpio.sh

#server options
DIST=0
REFRESH=1m

#setup parameters for selfplay
SV=800             # mcts simulations
G=32768            # games per net
OPT=0              # Optimizer 0=SGD 1=ADAM
LR=0.01            # learning rate
EPOCHS=1           # Number of epochs
NREPLAY=500000     # Number of games in the replay buffer
NSTEPS=235         # Number of steps
CPUCT=150          # Cpuct constant
POL_TEMP=100       # Policy temeprature
NOISE_FRAC=25      # Fraction of Dirchilet noise
POL_GRAD=0         # Use policy gradient algo.

#Network parameters
BOARDX=8
BOARDY=8
CHANNELS=24
POL_STYLE=1
NPOLICY=4672
NOAUXINP=
TRNFLG=--epd

#nets directory
NETS_DIR=${PWD}/nets

#kill background processes on exit
trap 'kill $(jobs -p)' EXIT

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
    rm -rf ${NETS_DIR}
    mkdir ${NETS_DIR}
    mkdir -p ${NETS_DIR}/hist
    mkdir -p ${NETS_DIR}/games
    mkdir -p ${NETS_DIR}/train
}

#convert to pb
convert-to-pb() {
    python src/k2tf_convert.py -m ${NETS_DIR}/$1 \
           --name $1.pb --prefix 'value' -n 2 -o ${NETS_DIR}
}

init() {
    python src/train.py --rand --nets $1 \
            ${NOAUXINP} --channels ${CHANNELS} --pol ${POL_STYLE} \
            --boardx ${BOARDX} --boardy ${BOARDY} --npolicy ${NPOLICY}
    convert-to-pb ID-0-model-$1
    if [ $GPUS -gt 0 ]; then
        convert-to-uff ${NETS_DIR}/ID-0-model-$1.pb -O value/Softmax -O policy/Reshape
        cp ${NETS_DIR}/ID-0-model-$1.uff ${NETS_DIR}/hist/ID-0-model-$1.uff
    fi
    cp ${NETS_DIR}/ID-0-model-$1 ${NETS_DIR}/hist/ID-0-model-$1
    cp ${NETS_DIR}/ID-0-model-$1.pb ${NETS_DIR}/hist/ID-0-model-$1.pb
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

if ! [ -e ${NETS_DIR} ]; then
    init0
    fornets init
fi

#which net format to use
Pnet=${net[0]}
if [ $GPUS -gt 0 ]; then
    NDIR=${NETS_DIR}/ID-0-model-${Pnet}.uff
else
    NDIR=${NETS_DIR}/ID-0-model-${Pnet}.pb
fi

#start network id
V=`find ${NETS_DIR}/hist/ID-*-model-${Pnet}.pb -type f | \
    sed 's/[ \t]*\([0-9]\{1,\}\).*/\1/' | grep -o [0-9]* | sort -rn | head -1`

#start server
send_server() {
   echo $@ > servinp
}

if [ $DIST -ge 1 ]; then
   echo "Starting server"
   if [ ! -p servinp ]; then
       mkfifo servinp
   fi
   tail -f servinp | java -cp nn-dist/bin ConsoleInterface -debug -startServer &
   sleep 5s
   send_server parameters ${SV} ${CPUCT} ${POL_TEMP} ${NOISE_FRAC}
   send_server network-uff ${NETS_DIR}/ID-0-model-${Pnet}.uff
   send_server network-pb ${NETS_DIR}/ID-0-model-${Pnet}.pb
   echo "Finished starting server"
else
   if [ ! -f ${SC}/${EXE} ]; then
       echo "Please set the correct path to " ${EXE}
       exit 0
   fi
fi

#run selfplay
run() {
    export CUDA_VISIBLE_DEVICES="$1" 
    SCOPT="reuse_tree 0 fpu_is_loss 0 fpu_red 0 cpuct_init ${CPUCT} \
           policy_temp ${POL_TEMP} noise_frac ${NOISE_FRAC}"
    taskset -c $3 time ./${EXE} nn_type 0 nn_path ${NDIR} new ${SCOPT} \
            sv ${SV} pvstyle 1 selfplayp $2 games$1.pgn train$1.epd quit
}

#use all gpus
rungames() {
    if [ $GPUS = 0 ]; then
        run 0 $1 0-$CPUS:1 &
    else
        I=$((CPUS/GPUS))
        GW=$((G/GPUS))
        for k in `seq 0 $((GPUS-1))`; do
            run $k $GW $((k*I))-$((k*I+I-1)):1 &
        done
    fi
    wait
    echo "All jobs finished"
}

#train network
train() {
    python src/train.py ${TRNFLG} ${NETS_DIR}/temp.epd --nets ${net[@]} --gpus ${GPUS} \
                --cores $((CPUS/2)) --opt ${OPT} --learning-rate ${LR} --epochs ${EPOCHS}  \
                --pol ${POL_STYLE} --pol_grad ${POL_GRAD} --channels ${CHANNELS} \
                --boardx ${BOARDX} --boardy ${BOARDY} --npolicy ${NPOLICY} ${NOAUXINP}
}

#move
move() {
    cp ${NETS_DIR}/ID-0-model-$1 ${NETS_DIR}/hist/ID-${V}-model-$1
    cp ${NETS_DIR}/ID-0-model-$1.pb ${NETS_DIR}/hist/ID-${V}-model-$1.pb
    if [ $GPUS -gt 0 ]; then
        cp ${NETS_DIR}/ID-0-model-$1.uff ${NETS_DIR}/hist/ID-${V}-model-$1.uff
    fi
}

#convert
conv() {
    E=`ls -l ${NETS_DIR}/ID-*-model-$1 | wc -l`
    E=$((E-1))
    mv ${NETS_DIR}/ID-$E-model-$1 ${NETS_DIR}/ID-0-model-$1
    rm -rf ${NETS_DIR}/ID-[1-$E]-model-$1
    convert-to-pb ID-0-model-$1
    if [ $GPUS -gt 0 ]; then
        convert-to-uff ${NETS_DIR}/ID-0-model-$1.pb -O value/Softmax -O policy/Reshape
        rm -rf ${NETS_DIR}/*.trt
    fi
}

#backup data
backup_data() {
    mv cgames.pgn ${NETS_DIR}/games/games$V.pgn
    gzip -f ${NETS_DIR}/games/games$V.pgn
    mv ctrain.epd ${NETS_DIR}/train/train$V.epd
    cp ${NETS_DIR}/train/train$V.epd ${NETS_DIR}/data$V.epd 
    gzip -f ${NETS_DIR}/train/train$V.epd
}

#get selfplay games
get_selfplay_games() {
    rm -rf cgames.pgn ctrain.epd
    cd ${SC}
    rungames ${G}
    cat games*.pgn > cgames.pgn
    cat train*.epd > ctrain.epd
    rm -rf games*.pgn train*.epd
    cd -
    mv ${SC}/cgames.pgn .
    mv ${SC}/ctrain.epd .
    backup_data
}

#get games from file
get_file_games() {
    while true; do
        sleep ${REFRESH}
        if [ -f ./ctrain.epd ]; then
            LN=`cat ctrain.epd | wc -l`
        else
            LN=0
        fi
        echo 'Accumulated games: ' $((LN/80)) of $G
        if [ $LN -ge $((G * 80)) ]; then
            echo 'Training new net'
            echo '----------------'
            backup_data
            return
        fi    
    done
}

#prepare training data
prepare() {
    
    #run games
    if [ $DIST -ge 1 ]; then
        send_server update-network
        get_file_games
    else
        get_selfplay_games
    fi

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
            rm -rf ${NETS_DIR}/data$i.epd
        done
    fi
    cat ${NETS_DIR}/data*.epd > ${NETS_DIR}/temp.epd

    shuf -n $((NSTEPS * 4096)) ${NETS_DIR}/temp.epd >x
    mv x ${NETS_DIR}/temp.epd
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
