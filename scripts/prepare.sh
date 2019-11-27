#!/bin/bash
NDIR=$1
SDIR=$( dirname ${BASH_SOURCE[0]} )
V=value/BiasAdd
P=policy/Reshape
if [ -z "$3" ]; then
   RN=$(seq 0 3)
else
   RN=$3
fi
for i in $RN; do
   $SDIR/convert-to-pb.sh ${NDIR}/ID-$2-model-$i; convert-to-uff $NDIR/ID-$2-model-$i.pb -O $V -O $P
done
if [ -z "$3" ]; then
  cp ${NDIR}/ID-$2-model-0.pb ${NDIR}/net-2x32.pb
  cp ${NDIR}/ID-$2-model-0.uff ${NDIR}/net-2x32.uff
  cp ${NDIR}/ID-$2-model-1.pb ${NDIR}/net-6x64.pb
  cp ${NDIR}/ID-$2-model-1.uff ${NDIR}/net-6x64.uff
  cp ${NDIR}/ID-$2-model-2.pb ${NDIR}/net-12x128.pb
  cp ${NDIR}/ID-$2-model-2.uff ${NDIR}/net-12x128.uff
  cp ${NDIR}/ID-$2-model-3.pb ${NDIR}/net-20x256.pb
  cp ${NDIR}/ID-$2-model-3.uff ${NDIR}/net-20x256.uff
fi