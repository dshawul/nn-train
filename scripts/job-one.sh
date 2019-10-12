#!/bin/bash

#number of cpus and gpus
CPUS=`grep -c ^processor /proc/cpuinfo`
if [ ! -z `which nvidia-smi` ]; then
    GPUS=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`
else
    GPUS=0
fi

#launch jobs
if [ $GPUS = 0 ]; then

   taskset -c 0-$CPUS:1 $@ &

else

for k in `seq 0 $((GPUS-1))`; do
   export CUDA_VISIBLE_DEVICES="$k" 
   I=$((CPUS/GPUS))
   B=$((k*I))-$((k*I+I-1)):1
   taskset -c $B $@ &
done

fi

#wait
wait
echo "All jobs finished on node: " `hostname`

