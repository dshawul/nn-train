#!/bin/bash
NDIR=$1
SDIR=$( dirname ${BASH_SOURCE[0]} )
if [ -z "$3" ]; then
   RN=$(seq 0 3)
else
   RN=$3
fi
for i in $RN; do
   $SDIR/convert-to-pb.sh ${NDIR}/ID-$2-model-$i ${NDIR}/infer-$i

   if [ $i -eq 0 ]; then
      cp ${NDIR}/ID-$2-model-0 ${NDIR}/net-2x32
      cp ${NDIR}/ID-$2-model-0.pb ${NDIR}/net-2x32.pb
      cp ${NDIR}/ID-$2-model-0.uff ${NDIR}/net-2x32.uff
      cp ${NDIR}/ID-$2-model-0.onnx ${NDIR}/net-2x32.onnx
   elif [ $i -eq 1 ]; then
      cp ${NDIR}/ID-$2-model-1 ${NDIR}/net-6x64
      cp ${NDIR}/ID-$2-model-1.pb ${NDIR}/net-6x64.pb
      cp ${NDIR}/ID-$2-model-1.uff ${NDIR}/net-6x64.uff
      cp ${NDIR}/ID-$2-model-1.onnx ${NDIR}/net-6x64.onnx
   elif [ $i -eq 2 ]; then
      cp ${NDIR}/ID-$2-model-2 ${NDIR}/net-12x128
      cp ${NDIR}/ID-$2-model-2.pb ${NDIR}/net-12x128.pb
      cp ${NDIR}/ID-$2-model-2.uff ${NDIR}/net-12x128.uff
      cp ${NDIR}/ID-$2-model-2.onnx ${NDIR}/net-12x128.onnx
   elif [ $i -eq 3 ]; then
      cp ${NDIR}/ID-$2-model-3 ${NDIR}/net-20x256
      cp ${NDIR}/ID-$2-model-3.pb ${NDIR}/net-20x256.pb
      cp ${NDIR}/ID-$2-model-3.uff ${NDIR}/net-20x256.uff
      cp ${NDIR}/ID-$2-model-3.onnx ${NDIR}/net-20x256.onnx
   elif [ $i -eq 4 ]; then
      cp ${NDIR}/ID-$2-model-4 ${NDIR}/net-20x384
      cp ${NDIR}/ID-$2-model-4.pb ${NDIR}/net-20x384.pb
      cp ${NDIR}/ID-$2-model-4.uff ${NDIR}/net-20x384.uff
      cp ${NDIR}/ID-$2-model-4.onnx ${NDIR}/net-20x384.onnx
   elif [ $i -eq 5 ]; then
      cp ${NDIR}/ID-$2-model-5 ${NDIR}/net-nnue
      cp ${NDIR}/ID-$2-model-5.pb ${NDIR}/net-nnue.pb
      cp ${NDIR}/ID-$2-model-5.uff ${NDIR}/net-nnue.uff
      cp ${NDIR}/ID-$2-model-5.onnx ${NDIR}/net-nnue.onnx
      cp ${NDIR}/ID-$2-model-5.bin ${NDIR}/net-nnue.bin
   fi
done
