#!/bin/bash

set -e

#set working directory and executable
SC=${PWD}/nn-dist/Scorpio/bin/Linux
EXE=scorpio.sh

#location of some external tools
CUTECHESS=~/cutechess-cli
BAYESELO=~/BayesElo/bayeselo

#server options
DIST=0
REFRESH=20s

#setup parameters for selfplay
SV=800             # mcts simulations
G=16384            # train net after this number of games
OPT=0              # Optimizer 0=SGD 1=ADAM
LR=0.2             # learning rate
EPOCHS=1           # Number of epochs
NREPLAY=$((32*G))  # Number of games in the replay buffer
NSTEPS=2560        # Number of steps
CPUCT=125          # Cpuct constant
POL_TEMP=120       # Policy temeprature
RAND_TEMP=80       # Temperature for random selection of moves
NOISE_FRAC=25      # Fraction of Dirchilet noise
POL_GRAD=0         # Use policy gradient algo.
POL_WEIGHT=2       # Policy weight
SCO_WEIGHT=1       # Score head weight
VAL_WEIGHT=1       # Value weight
RSAVO=8192         # Save weights with optimization after this many chunks
FRAC_PI=1          # Fraction of MCTS policy (PI) relative to one-hot policy(P)
FRAC_Z=1           # Fraction of ouctome(Z) relative to MCTS value(Q)

#Network parameters
BOARDX=8
BOARDY=8
CHANNELS=32
POL_CHANNELS=16
NOAUXINP=
TRNFLGS="--mixed"
NBATCH=512
BATCH_SIZE=512
DISTILL=0
PIECE_MAP="KQRBNPkqrbnp"
HEAD_TYPE=0

if [ $DISTILL != 0 ]; then
  FRAC_Z=0
  FRAC_PI=1
fi

#nets directory
WORK_ID=6
NETS_DIR=${PWD}/nets-${WORK_ID}

#pgn/epd source directory, and starting index
SRCPGN_DIR=files.txt
SRCEPD_DIR=${HOME}/storage/scorpiozero/nets-7/train
CI=0

#mpi
RANKS=1
SDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ $RANKS -gt 1 ]; then
   MPICMD="mpirun -np ${RANKS} --map-by node --bind-to none"
else
   MPICMD=
fi

#total steps
GLOBAL_STEPS=0

#kill background processes on exit
trap 'pkill -P $$' EXIT INT KILL TERM HUP

#number of cpus and gpus
CPUS=`grep -c ^processor /proc/cpuinfo`
if [ ! -z `which nvidia-smi` ]; then
    GPUS=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`
else
    GPUS=0
fi

##############
#display help
##############
display_help() {
    echo "Usage: $0 [Option...] {IDs} " >&2
    echo
    echo "   IDs           Network IDs to train 0..4 for 2x32,6x64,12x128,20x256,40x356."
    echo "   -h,--help     Display this help message."
    echo "   -m,--match    Conduct matches for evaluating networks. e.g. --match 0 200 201"
    echo "                 match 2x32 nets ID 200 and 201"
    echo "   -e,--elo      Calculate elos of networks."
    echo
}

if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    display_help
    exit 0
fi

############################
#calcuate elos of matches
############################
calculate_elo() {
   cat ${NETS_DIR}/matches/*.pgn > games.pgn
   (${BAYESELO} <<ENDM
      readpgn games.pgn
      elo
      mm
      covariance
      ratings
ENDM
   ) | grep scorpio | sed 's/scorpio/ID/g' > ${NETS_DIR}/ratings.txt

   cat ${NETS_DIR}/ratings.txt | sed 's/ID-//g' | \
	   awk '{ print $2 " " $3 " " $4 " " $5 }' | sort -rn | \
	   awk -v G=$((G/1024)) '{ print "{ x: "G*$1", y: ["$2+$3", "$2-$3"] }," }' \
	   > ${NETS_DIR}/pltdata
}

if [ "$1" == "-e" ] || [ "$1" == "--elo" ]; then
    calculate_elo
    exit 0
fi

###################
#conduct matches
###################
conduct_match() {
    ND1=${NETS_DIR}/hist/ID-$3-model-$1
    ND2=${NETS_DIR}/hist/ID-$2-model-$1

    if [ ! -f "$ND1" ] || [ ! -f "$ND2" ]; then
       exit 0
    fi

    if [ ! -f "$ND1.pb" ]; then
       ./scripts/prepare.sh ${NETS_DIR}/hist $3 $1 > /dev/null 2>&1
    fi
    if [ ! -f "$ND2.pb" ]; then
       ./scripts/prepare.sh ${NETS_DIR}/hist $2 $1 > /dev/null 2>&1
    fi

    if [ $GPUS -gt 0 ]; then
       ND1=${ND1}.uff
       ND2=${ND2}.uff
    else
       ND1=${ND1}.pb
       ND2=${ND2}.pb
    fi

    cd $CUTECHESS
    rm -rf match.pgn
    ./cutechess-cli -concurrency 1 \
        -engine cmd=${SC}/scorpio.sh dir=${SC} proto=xboard \
		arg="sv 8000 nn_type 0 nn_path ${ND1} alphabeta_man_c 0 float_type HALF" name=scorpio-$3 \
        -engine cmd=${SC}/scorpio.sh dir=${SC} proto=xboard \
		arg="sv 8000 nn_type 0 nn_path ${ND2} alphabeta_man_c 0 float_type HALF" name=scorpio-$2 \
        -each tc=40/30000 -rounds $4 -pgnout match.pgn -openings file=2moves.pgn \
	        format=pgn order=random -repeat
    cd - > /dev/null 2>&1

    cat ${CUTECHESS}/match.pgn >> ${NETS_DIR}/matches/match$2-$3.pgn
    rm -rf ${CUTECHESS}/match.pgn
}

if [ "$1" == "-m" ] || [ "$1" == "--match" ]; then
    conduct_match $2 $3 $4 100
    calculate_elo
    exit 0
fi

###################
#training
###################

#initialize training directory
init0() {
    echo "Initializing training."
    rm  -rf  ${NETS_DIR}
    mkdir -p ${NETS_DIR}
    mkdir -p ${NETS_DIR}/hist
    mkdir -p ${NETS_DIR}/games
    mkdir -p ${NETS_DIR}/train
    mkdir -p ${NETS_DIR}/matches
    touch ${NETS_DIR}/pltdata
    touch ${NETS_DIR}/description.txt
}

#initialize random network
init() {
    python src/train.py --rand --dir ${NETS_DIR} --nets $1 --batch-size ${BATCH_SIZE} \
            ${NOAUXINP} --channels ${CHANNELS} --policy-channels ${POL_CHANNELS} \
            --boardx ${BOARDX} --boardy ${BOARDY} --head-type ${HEAD_TYPE}
    ./scripts/prepare.sh ${NETS_DIR} 0 $1 > /dev/null 2>&1
    cp ${NETS_DIR}/ID-0-model-$1 ${NETS_DIR}/hist/ID-0-model-$1
    ln -sf ${NETS_DIR}/infer-$1 ${NETS_DIR}/hist/infer-$1
}

#loop over all nets
fornets() {
    for i in "${net[@]}"; do
        if ! [ -z $i ]; then
            $1 $i
        fi
    done
}

#clumsy network parsing
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
if [ $DIST -eq 3 ]; then
   V=$CI
else
   V=`ls -at ${NETS_DIR}/hist/ID-*-model-${Pnet} | head -1 | xargs -n 1 basename | grep -o -E '[0-9]+' | head -1`
fi

#start server
send_server() {
   echo $@ > servinp
}

if [ $DIST -eq 1 ]; then
   echo "Starting server"
   if [ ! -p servinp ]; then
       mkfifo servinp
   fi
   tail -f servinp | nn-dist/server.sh &
   sleep 5s
   send_server parameters ${WORK_ID} ${SV} ${CPUCT} ${POL_TEMP} ${NOISE_FRAC} ${HEAD_TYPE} ${RAND_TEMP}
   send_server network-uff ${NETS_DIR}/ID-0-model-${Pnet}.uff \
        "http://scorpiozero.ddns.net/scorpiozero/nets-${WORK_ID}/ID-0-model-${Pnet}.uff"
   echo "Finished starting server"
else
   if [ ! -f ${SC}/${EXE} ]; then
       echo "Please set the correct path to " ${EXE}
       exit 0
   fi
fi

#run selfplay
rungames() {
    if [ $GPUS = 0 ]; then
        GW=$(($1/RANKS))
    else
        GW=$(($1/(RANKS*GPUS)))
    fi
    if [ $DISTILL = 0 ]; then
        NETW="nn_type 0 nn_path ${NDIR}"
    else
        NETW=""
    fi
    SCOPT="train_data_type ${HEAD_TYPE} alphabeta_man_c 0 min_policy_value 0 \
           reuse_tree 0 fpu_is_loss 0 fpu_red 0 cpuct_init ${CPUCT} \
           backup_type 6 rand_temp ${RAND_TEMP} policy_temp ${POL_TEMP} noise_frac ${NOISE_FRAC}"
    ALLOPT="${NETW} new ${SCOPT} sv ${SV} \
	   pvstyle 1 selfplayp ${GW} games.pgn train.epd quit"
    time ${MPICMD} ./${EXE} ${ALLOPT}
}

#train network
train() {
    python src/train.py ${TRNFLGS} \
       --dir ${NETS_DIR} --epd ${NETS_DIR}/temp.epd --nets ${net[@]} --gpus ${GPUS} --cores $((CPUS/2)) --rsavo ${RSAVO} \
       --opt ${OPT} --learning-rate ${LR} --epochs ${EPOCHS} --piece-map ${PIECE_MAP} \
       --policy-weight ${POL_WEIGHT} --value-weight ${VAL_WEIGHT} --score-weight ${SCO_WEIGHT} \
       --policy-gradient ${POL_GRAD} --channels ${CHANNELS} --nbatch ${NBATCH} --batch-size ${BATCH_SIZE} --global-steps ${GLOBAL_STEPS} \
       --boardx ${BOARDX} --boardy ${BOARDY} --policy-channels ${POL_CHANNELS} --frac-pi ${FRAC_PI} --frac-z ${FRAC_Z} --head-type ${HEAD_TYPE} \
       ${NOAUXINP}
}

#move
move() {
    cp ${NETS_DIR}/ID-0-model-$1 ${NETS_DIR}/hist/ID-${V}-model-$1
}

#convert
conv() {
    E=`ls -l ${NETS_DIR}/ID-*-model-$1 | wc -l`
    E=$((E-1))

    #overwrite assuming it will pass
    mv ${NETS_DIR}/ID-$E-model-$1 ${NETS_DIR}/ID-0-model-$1
    for i in $(seq 1 $E); do
       rm -rf ${NETS_DIR}/ID-$i-model-$1
    done
    ./scripts/prepare.sh ${NETS_DIR} 0 $1 > /dev/null 2>&1
}

#backup data
backup_data() {
    mv cgames.pgn ${NETS_DIR}/games/games$V.pgn
    gzip -f ${NETS_DIR}/games/games$V.pgn
    cp ${NETS_DIR}/data$V.epd ${NETS_DIR}/train/train$V.epd
    gzip -f ${NETS_DIR}/train/train$V.epd
}

#get selfplay games
get_selfplay_games() {
    if [ $GPUS -gt 0 ]; then
        rm -rf ${NETS_DIR}/*.trt
    fi
    rm -rf cgames.pgn ctrain.epd
    cd ${SC}
    rungames ${G}
    cat games*.pgn* > cgames.pgn
    cat train*.epd* > ctrain.epd
    rm -rf games*.pgn* train*.epd*
    cd - > /dev/null 2>&1
    mv ${SC}/cgames.pgn .
    mv ${SC}/ctrain.epd .
}

#get games from file
get_file_games() {
    PLN=0
    while true; do
        sleep ${REFRESH}
        if [ -f ./cgames.pgn ]; then
            LN=`grep "Result" cgames.pgn | wc -l`
        else
            LN=0
        fi
        if [ $PLN -ne $LN ]; then
            echo 'Accumulated games: ' $LN of $G
            PLN=$LN
        fi
        if [ $LN -ge $G ]; then
            return
        fi    
    done
}

#train from PGN source
if [ -d ${SRCPGN_DIR}  ]; then
    SRCTYPE=0
    pgns=( `find ${SRCPGN_DIR} -type f` )
elif [ -f ${SRCPGN_DIR} ]; then
    SRCTYPE=1
    pgns=( `cat ${SRCPGN_DIR}` )
fi

#get fixed number of games from source
get_src_pgn() {
    T=0
    while true; do
        PGN=${pgns[${CI}]}
        PGNF=pgndir
        if ! [ -e ${PGNF} ]; then
            mkdir ${PGNF}
            if [ $SRCTYPE -eq 0 ]; then
                tar -xf ${PGN} -C ${PGNF} --strip-components=1
            elif [ $SRCTYPE -eq 1 ]; then
                wget ${PGN} -O xtempx
                tar -xf xtempx -C ${PGNF} --strip-components=1
                rm -rf xtempx
            fi
        else
            echo 'Skipping extraction.'
        fi

        P=$((G-T))
        M=`find ${PGNF} -type f | wc -l`
        if [ $M -lt $P ]; then
            P=$M
        fi
        F=`find ${PGNF} -type f | head -n ${P}`
        cat $F >>cgames.pgn
        rm -rf $F
        T=$((T+P))
        M=`find ${PGNF} -type f | wc -l`
        if [ $M -eq 0 ]; then
            CI=$((CI+1))
            rm -rf ${PGNF}
        fi

        echo "CI = " $CI
        echo "Games accumulated = " $T
        if [ $T -ge $G ]; then
            break
        fi
    done

    cp cgames.pgn ${SC}
    ${SC}/${EXE} pgn_to_epd cgames.pgn ctrain.epd quit
    mv ${SC}/ctrain.epd .
}

#get training positions from source
get_src_epd() {
    cp ${SRCEPD_DIR}/train$CI.epd.gz ${NETS_DIR}/data$CI.epd.gz
    gzip -fd ${NETS_DIR}/data$CI.epd.gz
    CI=$((CI+1))
}

#calcuate global steps
calc_global_steps() {
    ND=$((NREPLAY/G))

    if [ "$ND" -le "$V" ]; then
        GLOBAL_STEPS=$(( (NSTEPS * (2*V-ND+1)) / 2 ))
    else
        GLOBAL_STEPS=$(( (NSTEPS*(V)*(V+1)) / (2*ND) ))
    fi

    echo "Global number of steps trained so far: " $GLOBAL_STEPS
}

#prepare shuffled replay buffer
replay_buffer() {
    ND=$((NREPLAY/G))

    if [ "$ND" -le "$V" ]; then
        A=`seq 0 $((V-ND))`
        for i in $A; do
            rm -rf ${NETS_DIR}/data$i.epd
        done
    fi

    rm -rf x
    for i in ${NETS_DIR}/data*.epd; do
        shuf -n $((NSTEPS * BATCH_SIZE / ND)) $i >>x
    done

    mv x ${NETS_DIR}/temp.epd
}

#prepare training data
prepare() {
    
    #run games
    if [ $DIST -eq 3 ]; then
        get_src_epd
    elif [ $DIST -eq 2 ]; then
        get_src_pgn
    elif [ $DIST -eq 1 ]; then
        send_server update-network
        get_file_games
    else
        get_selfplay_games
    fi

    #backup data
    mv ctrain.epd ${NETS_DIR}/data$V.epd
    if [ $DIST -ne 3 ] && [ $DISTILL -eq 0 ]; then
       echo "Backing up training data"
       time backup_data
    fi

    #replay
    echo "Sampling from replay buffer"
    time replay_buffer
}

#Selfplay training loop
selfplay_loop() {
    while true ; do
        echo 'Collecting games'
        time prepare

        calc_global_steps

        SCPID=`pidof scorpio`
        $( kill -STOP ${SCPID} ) || true

        echo 'Training new net from net ID = ' $V
        time train

        echo 'Converting nets to UFF'
        time fornets conv

        $( kill -CONT ${SCPID} ) || true

        V=$((V+1))

        fornets move

    done
}

selfplay_loop
