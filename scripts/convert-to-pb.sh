#!/bin/bash
SDIR=$( dirname ${BASH_SOURCE[0]} )
if [ -z "$2" ]; then
    IM=
else
    IM="-i $2"
fi
python $SDIR/../src/convert-to-pb.py -m $1 $IM
