#!/bin/bash

set -e

#setup parameters for selfplay
SC=~/Scorpio   # path to scorpio exec
NREPLAY=500000 # Number of games in the replay buffer
NSTEPS=250     # Number of steps
NGAMES=32000   # Number of games for training network

#display help
display_help() {
    echo "Usage: $0 [Option...] {file} {strt} " >&2
    echo
    echo "   -h     Display this help message."
    echo " file     File contaiing list of tar files."
    echo " strt     Start ID of file."
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

# Prepare training data from tar file pgns
CI=0
if ! [ -z "$2" ]; then 
    CI=$2
fi

declare -a pgns=( `cat $1` )

get_lczero_games() {
    T=0
    while true; do
        PGN=${pgns[${CI}]}
        PGNF=${PGN%%.*}
        if ! [ -e ${PGNF} ]; then
            wget http://data.lczero.org/files/${PGN}
            tar -xf ${PGN}
            rm -rf ${PGN}
        else
            echo 'Skipping download'
        fi

        P=$((32000-T))
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
        if [ $T -ge 32000 ]; then
            break   
        fi
    done
}

prepare_train_data() {
    
    V=`find lcdata/data*.epd -type f | grep -o [0-9]* | sort -rn | head -1`
    V=$((V+1))

    while true ; do
        rm -rf cgames.pgn

        get_lczero_games

        ${SC}/scorpio pgn_to_epd cgames.pgn nets/data$V.epd quit

        ND=$((NREPLAY/32000))
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

        cp nets/temp.epd lcdata/data$V.epd

        V=$((V+1))
    done
}

prepare_train_data
