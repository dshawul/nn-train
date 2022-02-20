#!/bin/bash

set -e

NET=5
HEAD_TYPE=3
BATCH_SIZE=8192
CHANNELS=12
NETS_DIR=nets-k46
MAX_STEPS=12800
SAVE_STEPS=32

mkdir -p ${NETS_DIR}/hist

#convert
conv() {
    E=`ls -l ${NETS_DIR}/ID-*-model-$1 | wc -l`
    E=$((E-1))
  
    GSTEPS=$((GSTEPS + E*SAVE_STEPS))

    #overwrite assuming it will pass
    mv ${NETS_DIR}/ID-$E-model-$1 ${NETS_DIR}/ID-0-model-$1
    for i in $(seq 1 $E); do
       rm -rf ${NETS_DIR}/ID-$i-model-$1
    done

    cp ${NETS_DIR}/ID-0-model-$1 ${NETS_DIR}/hist/ID-$2-model-$1
}

ID=0
START=0
EPOCHS=700
FACT=0.99
LR_INIT=0.2
GSTEPS=$((START*MAX_STEPS))

#training loop
for (( EP=$START; EP<$EPOCHS; EP++ )); do
    LR=`awk "BEGIN { print $LR_INIT*($FACT^$EP) }"`
    echo "Epoch" $EP with "learning rate:" $LR
    #continue

    #training data file
    DATA_DIR=/home/daniel/storage/nnue-data
    TAG=$( printf %03d $EP )
    DATA=$DATA_DIR/temp.epd.$TAG.gz

    echo "Data file:" $DATA

    #train
    time python src/train.py --opt 0 --no-auxinp --gpus 1 --cores 12 --rsav $SAVE_STEPS --rsavo $MAX_STEPS --frac-z 0 \
                        --id $ID --max-steps $MAX_STEPS --global-steps $GSTEPS --dir $NETS_DIR --gzip --epd ${DATA} \
                        --head-type $HEAD_TYPE --channels $CHANNELS --net $NET \
                        --learning-rate $LR --batch-size $BATCH_SIZE
    
    conv $NET $EP 
done
