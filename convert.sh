#!/bin/bash
python k2tf_convert.py -m nets/$1 --name $1.pb --prefix 'value' -n 1 -o nets/
