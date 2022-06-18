#!/bin/bash

set -e

################################
# Training mode
#    0 = local selfplay
#    1 = server-client selfplay
#    2 = supervized
################################

TRAIN_MODE=1

# Use a fixed net in selfplay modes
DISTILL=0

##############
# directories
##############

# directory to scorpio binary
SC=${PWD}/nn-dist/Scorpio/bin/Linux
EXE=scorpio.sh

# network directory
WORK_ID=14
NETS_DIR=${HOME}/storage/scorpiozero/nets-$(printf "%02d" ${WORK_ID})

# For supervized training, set source data directory of gzipped epd files,
# or a file containing a list of them. CI is the index indicating where to
# to start training.
SRC_DATA_DIR=${HOME}/storage/train-data
CI=0

# location of some external tools
CUTECHESS=${HOME}/cutechess-cli
BAYESELO=${HOME}/BayesElo/bayeselo

##############
# settings
##############

# setup parameters for selfplay and training
MONTECARLO=1          # Use montecarlo search for selfplay
SV=800                # simulations limit for MCTS
SD=4                  # depth limit for AB search
OPT=0                 # Optimizer 0=SGD 1=ADAM
LR=0.2                # learning rate
NGAMES=24576          # train net after this number of games
BATCH_SIZE=1024       # Mini-batch size
NSTEPS=1280           # Number of steps to train for
NREPLAY=32            # Games in replay buffer = (NREPLAY * NGAMES)
SAVE_STEPS=256        # Save network after this many steps
SAVE_STEPS_O=$NSTEPS  # Save optimizer state steps (=0 means don't save optimizer state)
CPUCT=125             # Cpuct constant
CPUCT_ROOT_FAC=100    # Mulitply Cpuct at root by this factor
POL_TEMP=110          # Policy temeprature
POL_TEMP_ROOT_FAC=100 # Multiply Policy temeprature at root by this factor
NOISE_FRAC=25         # Fraction of Dirchilet noise
NOISE_ALPHA=30        # Alpha parameter
NOISE_BETA=100        # Beta parameter
TEMP_PLIES=30         # Number of plies to apply for noise
RAND_TEMP=90          # Temperature for random selection of moves
RAND_TEMP_DELTA=0     # Decrease temperature linearly by this much
RAND_TEMP_END=0       # Endgame temperature for random selection of moves
POL_GRAD=0            # Use policy gradient algo.
POL_WEIGHT=2          # Policy weight
SCO_WEIGHT=1          # Score head weight
VAL_WEIGHT=1          # Value weight
FRAC_PI=1             # Fraction of MCTS policy (PI) relative to one-hot policy(P)
FRAC_Z=1              # Fraction of ouctome(Z) relative to MCTS value(Q)
FORCED_PLAYOUTS=0     # Forced playouts
POLICY_PRUNING=0      # Policy pruning
FPU_IS_LOSS=0         # FPU is loss,win or reduction
FPU_RED=33            # FPU reduction level
PLAYOUT_CAP=0         # Playout cap randomization
FRAC_FULL_PLAY=25     # Fraction of positions where full playouts are used
FRAC_SV_LOW=30        # Fraction of visits for the low playouts
RESIGN=600            # Resign value

# Network parameters
HEAD_TYPE=0              # output head type
BOARDX=8                 # Board dimension in X
BOARDY=8                 # Board dimension in Y
CHANNELS=32              # Number of input channels
POL_CHANNELS=16          # Number of policy channels
PIECE_MAP="KQRBNPkqrbnp" # Piece characters

#Additional training flags
TRNFLGS="--mixed"

# server refresh rate
REFRESH=20s

# Adjust Z and PI ratio for distillation
if [ $DISTILL != 0 ]; then
    FRAC_Z=0
    FRAC_PI=1
fi

##############
# hardware
##############

# mpi
RANKS=1
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ $RANKS -gt 1 ]; then
    MPICMD="mpirun -np ${RANKS} --map-by node --bind-to none"
else
    MPICMD=
fi

# kill background processes on exit
trap 'pkill -P $$' EXIT INT KILL TERM HUP

# number of cpus and gpus
CPUS=$(grep -c ^processor /proc/cpuinfo)
if [ ! -z $(which nvidia-smi) ]; then
    GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    GPUS=0
fi

# time a command
time_command() {
    echo "Executing: $@"
    local start=$(date +%s)
    "$@"
    local end=$(date +%s)
    echo "Finished: $@ in" $((end - start)) "sec."
}

##############
# display help
##############
display_help() {
    echo "Usage: $0 [Option...] {IDs} " >&2
    echo
    echo "   ID            Network ID to train 0..4 for 2x32,6x64,12x128,20x256,40x356."
    echo "   -h,--help     Display this help message."
    echo "   -m,--match    Conduct matches for evaluating networks. e.g. --match 0 200 201"
    echo "                 match 2x32 nets ID 200 and 201"
    echo "   -e,--elo      Calculate elos of networks."
    echo
}

if [ $# -eq 0 ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    display_help
    exit 0
fi

############################
# calcuate elos of matches
############################
calculate_elo() {
    cat ${NETS_DIR}/matches/*.pgn >games.pgn
    (
        ${BAYESELO} <<ENDM
           readpgn games.pgn
           elo
           mm
           covariance
           ratings
ENDM
    ) | grep scorpio | sed 's/scorpio/ID/g' >${NETS_DIR}/ratings.txt

    cat ${NETS_DIR}/ratings.txt | sed 's/ID-//g' |
        awk '{ print $2 " " $3 " " $4 " " $5 }' | sort -rn |
        awk -v G=$((NGAMES / 1024)) '{ print "{ x: "G*$1", y: ["$2+$3", "$2-$3"] }," }' \
            >${NETS_DIR}/pltdata
}

if [ "$1" == "-e" ] || [ "$1" == "--elo" ]; then
    calculate_elo
    exit 0
fi

###################
# conduct matches
###################
conduct_match() {
    ND1=${NETS_DIR}/hist/ID-$3-model-$1
    ND2=${NETS_DIR}/hist/ID-$2-model-$1

    if [ ! -f "$ND1" ] || [ ! -f "$ND2" ]; then
        exit 0
    fi

    if [ ! -f "$ND1.pb" ]; then
        ./scripts/prepare.sh ${NETS_DIR}/hist $3 $1 >/dev/null 2>&1
    fi
    if [ ! -f "$ND2.pb" ]; then
        ./scripts/prepare.sh ${NETS_DIR}/hist $2 $1 >/dev/null 2>&1
    fi

    if [ $1 = 5 ]; then
        ND1=${ND1}.bin
        ND2=${ND2}.bin
    else
        ND1=${ND1}.uff
        ND2=${ND2}.uff
    fi

    cd $CUTECHESS
    rm -rf match.pgn

    if [ $1 = 5 ]; then
        MATCH_OPTS="montecarlo 0 mt 1 use_nn 0 use_nnue 1 nnue_type 1 nnue_scale 128 nnue_path"
        CONCUR=10
        TC=20+0.5
    else
        MATCH_OPTS="montecarlo 1 sv 4000 alphabeta_man_c 0 float_type HALF use_nn 1 use_nnue 0 nn_type 0 nn_path"
        CONCUR=1
        TC=40/80000
    fi

    ./cutechess-cli -concurrency $CONCUR -resign movecount=3 score=500 \
        -engine cmd=${SC}/scorpio.sh dir=${SC} proto=xboard \
        arg="${MATCH_OPTS} ${ND1}" name=scorpio-$3 \
        -engine cmd=${SC}/scorpio.sh dir=${SC} proto=xboard \
        arg="${MATCH_OPTS} ${ND2}" name=scorpio-$2 \
        -each tc=$TC -rounds $4 -pgnout match.pgn -openings file=2moves.pgn \
        format=pgn order=random -repeat

    cd - >/dev/null 2>&1

    cat ${CUTECHESS}/match.pgn >>${NETS_DIR}/matches/match$2-$3.pgn
    rm -rf ${CUTECHESS}/match.pgn
}

if [ "$1" == "-m" ] || [ "$1" == "--match" ]; then
    conduct_match $2 $3 $4 200
    calculate_elo
    exit 0
fi

#########################
# initialization
#########################

# initialize training directory
init0() {
    echo "Initializing training."
    rm -rf ${NETS_DIR}
    mkdir -p ${NETS_DIR}
    mkdir -p ${NETS_DIR}/hist
    mkdir -p ${NETS_DIR}/games
    mkdir -p ${NETS_DIR}/train
    mkdir -p ${NETS_DIR}/matches
    touch ${NETS_DIR}/pltdata
    touch ${NETS_DIR}/description.txt
}

# initialize random network
init() {
    python -W ignore src/train.py ${TRNFLGS} --rand \
        --dir ${NETS_DIR} --net $1 --batch-size ${BATCH_SIZE} \
        --channels ${CHANNELS} --policy-channels ${POL_CHANNELS} \
        --boardx ${BOARDX} --boardy ${BOARDY} --head-type ${HEAD_TYPE}

    ./scripts/prepare.sh ${NETS_DIR} 0 $1 >/dev/null 2>&1
    cp ${NETS_DIR}/ID-0-model-$1 ${NETS_DIR}/hist/ID-0-model-$1
    ln -sf ${NETS_DIR}/infer-$1 ${NETS_DIR}/hist/infer-$1
}

# network id
NID=$1

# wrapper
fornets() {
    $1 $NID
}

# init
if ! [ -e ${NETS_DIR} ]; then
    init0
    fornets init
fi

# which net format to use
if [ $NID -eq 5 ]; then
    NEXT=bin
else
    NEXT=uff
fi
NDIR=${NETS_DIR}/ID-0-model-${NID}.${NEXT}

# start network id
V=$(ls -at ${NETS_DIR}/hist/ID-*-model-${NID} | head -1 | xargs -n 1 basename | grep -o -E '[0-9]+' | head -1)

#########################
# generate training data
#########################

# options for Scorpio
if [ $MONTECARLO = 0 ]; then
    SCOPT="montecarlo 0 filter_quiet 1 \
       sp_resign_value ${RESIGN} train_data_type ${HEAD_TYPE} sd ${SD} \
       temp_plies ${TEMP_PLIES} rand_temp ${RAND_TEMP} rand_temp_delta ${RAND_TEMP_DELTA} rand_temp_end ${RAND_TEMP_END}"
else
    SCOPT="montecarlo 1 early_stop 0 reuse_tree 0 backup_type 6 alphabeta_man_c 0 min_policy_value 0 \
       sp_resign_value ${RESIGN} train_data_type ${HEAD_TYPE} sv ${SV} \
       playout_cap_rand ${PLAYOUT_CAP} frac_full_playouts ${FRAC_FULL_PLAY} frac_sv_low ${FRAC_SV_LOW} \
       forced_playouts ${FORCED_PLAYOUTS} policy_pruning ${POLICY_PRUNING} \
       fpu_is_loss ${FPU_IS_LOSS} fpu_red ${FPU_RED} \
       cpuct_init ${CPUCT} cpuct_init_root_factor ${CPUCT_ROOT_FAC} \
       policy_temp ${POL_TEMP} policy_temp_root_factor ${POL_TEMP_ROOT_FAC} \
       temp_plies ${TEMP_PLIES} rand_temp ${RAND_TEMP} rand_temp_delta ${RAND_TEMP_DELTA} rand_temp_end ${RAND_TEMP_END} \
       noise_frac ${NOISE_FRAC} noise_alpha ${NOISE_ALPHA} noise_beta ${NOISE_BETA}"
fi
if [ $DISTILL = 0 ]; then
    if [ $NID -eq 5 ]; then
        NOPTS="use_nn 0 use_nnue 1 nnue_type 1 nnue_scale 128 nnue_path"
    else
        NOPTS="use_nn 1 use_nnue 0 nn_type 0 nn_path"
    fi
    if [ $TRAIN_MODE -eq 1 ]; then
        SCOPT="${NOPTS} ../../../net.uff new ${SCOPT}"
    else
        SCOPT="${NOPTS} ${NDIR} new ${SCOPT}"
    fi
fi

# start server
send_server() {
    echo $@ >servinp
}

if [ $TRAIN_MODE -eq 1 ]; then
    echo "Starting server"
    if [ ! -p servinp ]; then
        mkfifo servinp
    fi
    tail -f servinp | nn-dist/server.sh &
    sleep 5s
    send_server parameters ${WORK_ID} ${SCOPT}
    send_server network-uff ${NDIR} \
        "http://scorpiozero.ddns.net/scorpiozero/nets-${WORK_ID}/ID-0-model-${NID}.${NEXT}"
    echo "Finished starting server"
else
    if [ ! -f ${SC}/${EXE} ]; then
        echo "Please set the correct path to " ${EXE}
        exit 0
    fi
fi

# run selfplay
rungames() {
    if [ $GPUS = 0 ]; then
        GW=$(($1 / RANKS))
    else
        GW=$(($1 / (RANKS * GPUS)))
    fi
    ALLOPT="${SCOPT} pvstyle 1 selfplayp ${GW} games.pgn train.epd quit"
    time_command ${MPICMD} ./${EXE} ${ALLOPT}
}

# get selfplay games
get_selfplay_games() {
    if [ $GPUS -gt 0 ]; then
        rm -rf ${NETS_DIR}/*.trt
    fi
    rm -rf cgames.pgn ctrain.epd
    cd ${SC}
    rungames ${NGAMES}
    cat games*.pgn* >cgames.pgn
    cat train*.epd* >ctrain.epd
    rm -rf games*.pgn* train*.epd*
    cd - >/dev/null 2>&1
    mv ${SC}/cgames.pgn .
    mv ${SC}/ctrain.epd .
}

# get games from file
get_file_games() {
    PLN=0
    while true; do
        sleep ${REFRESH}
        if [ -f ./cgames.pgn ]; then
            LN=$(grep Result cgames.pgn | wc -l)
        else
            LN=0
        fi
        if [ $PLN -ne $LN ]; then
            PP=$(wc -l ctrain.epd | awk '{print $1}')
            echo "Accumulated: games = $LN of $NGAMES, and positions = $PP of $((NSTEPS * BATCH_SIZE)), " \
                "sampling ratio $(((LN * NSTEPS * BATCH_SIZE * 100) / (PP * NGAMES)))%"
            PLN=$LN
        fi
        if [ $LN -ge $NGAMES ]; then
            return
        fi
    done
}

###################
# Data formats
###################

# data stream
if [ $TRAIN_MODE -ge 2 ]; then
    if [ -d ${SRC_DATA_DIR} ]; then
        data_stream=($(ls ${SRC_DATA_DIR}/*.gz))
    elif [ -f ${SRC_DATA_DIR} ]; then
        data_stream=($(cat ${SRC_DATA_DIR}))
    fi
fi

# get training epd positions
get_src_epd() {
    rm -rf ctrain.epd
    while true; do
        EPD=${data_stream[${CI}]}
        echo "CI =" $CI $EPD
        cp ${EPD} ${NETS_DIR}/current.epd.gz
        gzip -fd ${NETS_DIR}/current.epd.gz
        cat ${NETS_DIR}/current.epd >>ctrain.epd
        rm -rf ${NETS_DIR}/current.epd
        CI=$((CI + 1))

        NPOS=$(wc -l ctrain.epd | awk '{print $1}')
        MAXN=$((NGAMES * 80))
        if [ $NPOS -ge $MAXN ]; then
            break
        fi
    done
}

###################
# Training
###################

# stop/resume scorpio
SCPID=

stop_scorpio() {
    SCPID=$(pidof scorpio) || true
    if [ ! -z ${SCPID} ]; then
        $(kill -STOP ${SCPID}) || true
    fi
}

resume_scorpio() {
    if [ ! -z ${SCPID} ]; then
        $(kill -CONT ${SCPID}) || true
    fi
}

# calcuate global steps
calc_global_steps() {
    if [ "$NREPLAY" -le "$V" ]; then
        GLOBAL_STEPS=$(((NSTEPS * (2 * V - NREPLAY + 1)) / 2))
    else
        GLOBAL_STEPS=$(((NSTEPS * (V) * (V + 1)) / (2 * NREPLAY)))
    fi
}

# prepare shuffled replay buffer
replay_buffer() {
    if [ "$NREPLAY" -le "$V" ]; then
        A=$(seq 0 $((V - NREPLAY)))
        for i in $A; do
            rm -rf ${NETS_DIR}/data$i.epd
        done
    fi

    rm -rf x
    for i in ${NETS_DIR}/data*.epd; do
        shuf -n $((NSTEPS * BATCH_SIZE / NREPLAY)) $i >>x
    done

    if [ $NREPLAY -gt 1 ]; then
        shuf x -o ${NETS_DIR}/temp.epd
        rm -rf x
    else
        mv x ${NETS_DIR}/temp.epd
    fi
}

# prepare training data
prepare() {

    #run games
    if [ $TRAIN_MODE -eq 2 ]; then
        get_src_epd
    elif [ $TRAIN_MODE -eq 1 ]; then
        send_server update-network
        get_file_games
    else
        get_selfplay_games
    fi

    stop_scorpio

    #backup data
    mv ctrain.epd ${NETS_DIR}/data$V.epd
    if [ $TRAIN_MODE -ne 2 ] && [ $DISTILL -eq 0 ]; then
        time_command backup_data
    fi

    stop_scorpio

    #replay
    time_command replay_buffer
}

# move
move() {
    cp ${NETS_DIR}/ID-0-model-$1 ${NETS_DIR}/hist/ID-${V}-model-$1
}

# convert
conv() {
    E=$(ls -l ${NETS_DIR}/ID-*-model-$1 | wc -l)
    E=$((E - 1))

    #overwrite assuming it will pass
    mv ${NETS_DIR}/ID-$E-model-$1 ${NETS_DIR}/ID-0-model-$1
    for i in $(seq 1 $E); do
        rm -rf ${NETS_DIR}/ID-$i-model-$1
    done
    ./scripts/prepare.sh ${NETS_DIR} 0 $1 >/dev/null 2>&1
}

# backup data
backup_data() {
    mv cgames.pgn ${NETS_DIR}/games/games$V.pgn
    gzip -f ${NETS_DIR}/games/games$V.pgn
    cp ${NETS_DIR}/data$V.epd ${NETS_DIR}/train/train$V.epd
    gzip -f ${NETS_DIR}/train/train$V.epd
}

# train network
train() {
    python -W ignore src/train.py ${TRNFLGS} \
        --dir ${NETS_DIR} --epd ${NETS_DIR}/temp.epd --net $NID --gpus ${GPUS} --cores $((CPUS / 2)) \
        --rsav ${SAVE_STEPS} --rsavo ${SAVE_STEPS_O} --opt ${OPT} --max-steps ${NSTEPS} --learning-rate ${LR} \
        --policy-weight ${POL_WEIGHT} --value-weight ${VAL_WEIGHT} --score-weight ${SCO_WEIGHT} \
        --policy-gradient ${POL_GRAD} --channels ${CHANNELS} --batch-size ${BATCH_SIZE} --global-steps ${GLOBAL_STEPS} \
        --boardx ${BOARDX} --boardy ${BOARDY} --policy-channels ${POL_CHANNELS} \
        --frac-pi ${FRAC_PI} --frac-z ${FRAC_Z} --head-type ${HEAD_TYPE} \
        --piece-map ${PIECE_MAP}
    echo
}

# Selfplay training loop
selfplay_loop() {
    while true; do
        calc_global_steps
        echo 'Network ID =' $V ', Number of steps =' $GLOBAL_STEPS

        time_command prepare
        stop_scorpio
        time_command train
        fornets conv
        resume_scorpio

        V=$((V + 1))
        fornets move
    done
}

selfplay_loop
