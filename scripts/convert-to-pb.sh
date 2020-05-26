#!/bin/bash
BNAME=$( basename $1 )
DNAME=$( dirname $1 )
NOUT=2
if [ -z "$2" ]; then
	IM=
else
	IM="-i $2"
fi
python src/k2tf_convert.py -m $1 $IM --name $BNAME.pb \
       --prefix 'value' -n $NOUT -o $DNAME
