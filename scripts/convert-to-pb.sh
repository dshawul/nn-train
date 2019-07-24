#!/bin/bash
BNAME=$( basename $1 )
DNAME=$( dirname $1 )
python src/k2tf_convert.py -m $1 --name $BNAME.pb \
       --prefix 'value' -n 2 -o $DNAME
