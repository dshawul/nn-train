#!/bin/bash
NDIR=$1
SDIR=$( dirname ${BASH_SOURCE[0]} )
V=value/Softmax
P=policy/Reshape
cp $NDIR/ID-$2-model-0 $NDIR/net-2x32; $SDIR/convert-to-pb.sh ${NDIR}/net-2x32; convert-to-uff $NDIR/net-2x32.pb -O $V -O $P
cp $NDIR/ID-$2-model-1 $NDIR/net-6x64; $SDIR/convert-to-pb.sh ${NDIR}/net-6x64; convert-to-uff $NDIR/net-6x64.pb -O $V -O $P
cp $NDIR/ID-$2-model-2 $NDIR/net-12x128; $SDIR/convert-to-pb.sh ${NDIR}/net-12x128; convert-to-uff $NDIR/net-12x128.pb -O $V -O $P
cp $NDIR/ID-$2-model-3 $NDIR/net-20x256; $SDIR/convert-to-pb.sh ${NDIR}/net-20x256; convert-to-uff $NDIR/net-20x256.pb -O $V -O $P
