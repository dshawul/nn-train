#!/bin/bash

set -e

#set working directory and executable
SC=${PWD}/nn-dist/Scorpio-train/bin/Linux
EXE=scorpio.sh

#location of some external tools
CUTECHESS=~/cutechess-cli
BAYESELO=~/BayesElo/bayeselo

#server options
DIST=0
REFRESH=20s

#setup parameters for selfplay
SV=1               # mcts simulations
G=16384            # train net after this number of games
OPT=0              # Optimizer 0=SGD 1=ADAM
LR=0.02            # learning rate
EPOCHS=1           # Number of epochs
NREPLAY=$((32*G))  # Number of games in the replay buffer
NSTEPS=256         # Number of steps
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
BATCH_SIZE=4096

#nets directory
WORK_ID=6
NETS_DIR=${PWD}/nets-${WORK_ID}

#kill background processes on exit
trap 'kill $(jobs -p)' EXIT INT

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
    MH=`find ${NETS_DIR}/hist/ID-*-model-$1.pb -type f | \
      sed 's/[ \t]*\([0-9]\{1,\}\).*/\1/' | grep -o [0-9]* | sort -rn | head -1`

    if [ $GPUS -gt 0 ]; then
       ND1=${NETS_DIR}/hist/ID-$3-model-$1.uff
       ND2=${NETS_DIR}/hist/ID-$2-model-$1.uff
    else
       ND1=${NETS_DIR}/hist/ID-$3-model-$1.pb
       ND2=${NETS_DIR}/hist/ID-$2-model-$1.pb
    fi

    if [ ! -f "$ND1" ] || [ ! -f "$ND2" ]; then
       exit 0
    fi

    cd $CUTECHESS
    rm -rf match.pgn
    ./cutechess-cli -concurrency 1 \
        -engine cmd=${SC}/scorpio.sh dir=${SC} proto=xboard \
		arg="sv 800 nn_type 0 nn_path ${ND1}" name=scorpio-$3 \
        -engine cmd=${SC}/scorpio.sh dir=${SC} proto=xboard \
		arg="sv 800 nn_type 0 nn_path ${ND2}" name=scorpio-$2 \
        -each tc=40/30000 -rounds $4 -pgnout match.pgn -openings file=2moves.pgn \
	        format=pgn order=random -repeat
    cd -

    cat ${CUTECHESS}/match.pgn >> ${NETS_DIR}/matches/match$2-$3.pgn
    rm -rf ${CUTECHESS}/match.pgn
}

if [ "$1" == "-m" ] || [ "$1" == "--match" ]; then
    conduct_match $2 $3 $4 200
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
}

#convert to pb
convert-to-pb() {
    python src/k2tf_convert.py -m ${NETS_DIR}/$1 \
           --name $1.pb --prefix 'value' -n 2 -o ${NETS_DIR}
}

#initialize random network
init() {
    python src/train.py --rand --dir ${NETS_DIR} --nets $1 --batch-size ${BATCH_SIZE} \
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
V=`ls -l ${NETS_DIR}/hist/ID-*-model-${Pnet}.pb | wc -l`
V=$((V-1))

#start server
send_server() {
   echo $@ > servinp
}

if [ $DIST -ge 1 ]; then
   echo "Starting server"
   if [ ! -p servinp ]; then
       mkfifo servinp
   fi
   tail -f servinp | nn-dist/server.sh &
   sleep 5s
   send_server parameters ${WORK_ID} ${SV} ${CPUCT} ${POL_TEMP} ${NOISE_FRAC}
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
    MG=$(((V+1)*G))
    if [ ${MG} -le ${NREPLAY} ]; then
	MLR=`echo "(${LR}*${MG})/${NREPLAY}" | bc -l`
    else
	MLR=$LR
    fi
    python src/train.py \
       --dir ${NETS_DIR} ${TRNFLG} ${NETS_DIR}/temp.epd --nets ${net[@]} --gpus ${GPUS} \
       --cores $((CPUS/2)) --opt ${OPT} --learning-rate ${MLR} --epochs ${EPOCHS}  \
       --pol ${POL_STYLE} --pol_grad ${POL_GRAD} --channels ${CHANNELS} --batch-size ${BATCH_SIZE} \
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

    #overwrite assuming it will pass
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
        if [ -f ./cgames.pgn ]; then
            LN=`grep "Result" cgames.pgn | wc -l`
        else
            LN=0
        fi
        echo 'Accumulated games: ' $LN of $G
        if [ $LN -ge $G ]; then
            echo '================'
            echo 'Training new net'
            echo '================'
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
    ND=$((NREPLAY/G))
    if [ $ND -ge $V ]; then
        ND=$((V+1))
    else
        A=`seq 0 $((V-ND-1))`
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

#Selfplay training loop
selfplay_loop() {
    while true ; do

        prepare

        train

        fornets conv

        V=$((V+1))

        fornets move

    done
}

selfplay_loop
