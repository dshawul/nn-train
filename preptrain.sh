#!/bin/bash

set -e

#setup parameters for selfplay
SC=~/Scorpio   # path to scorpio exec
NREPLAY=500000 # Number of games in the replay buffer
NSTEPS=250     # Number of steps of training
NGAMES=32000   # Number of games for training one network
EPDDIR=epd     # Directory to store epd files for training

#display help
display_help() {
    echo "Usage: $0 [Options...] " >&2
    echo
    echo "   -h                Display this help message."
    echo "   -d [dir]  {ID}    Source is directory [dir] -- start at file number {ID}."
    echo "   -w [file] {ID}    Source is [file] containing links to pgn tar files -- start at file number {ID}."
    echo
}

#options
if [ "$1" == "-h" ]; then
    display_help
    exit 0
elif [ "$1" == "-d" ]; then
    SRCTYPE=0
    pgns=( `find $2 -type f` )
elif [ "$1" == "-w" ]; then
    SRCTYPE=1
    pgns=( `cat $2` )
fi

CI=0
if ! [ -z "$3" ]; then 
    CI=$3
fi

#check if Scorpio directory exists
if [ ! -f ${SC}/scorpio ]; then
    echo "Please set the correct path to scorpio."
    exit 0
fi

# get fixed number of games from source
get_games() {
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

        P=$((NGAMES-T))
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
        if [ $T -ge $NGAMES ]; then
            break   
        fi
    done
}

# Prepare training data from tar file pgns
prepare_train_data() {
    
    if [ -e ${EPDDIR} ]; then
        V=`find ${EPDDIR}/data*.epd -type f | grep -o [0-9]* | sort -rn | head -1`
        V=$((V+1))
    else
        mkdir ${EPDDIR}
        V=0
    fi

    while true ; do
        rm -rf cgames.pgn

        get_games

        ${SC}/scorpio pgn_to_epd cgames.pgn nets/data$V.epd quit

        ND=$((NREPLAY/NGAMES))
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

        cp nets/temp.epd ${EPDDIR}/data$V.epd

        V=$((V+1))
    done
}

prepare_train_data
