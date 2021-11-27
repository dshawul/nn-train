#!/bin/sh
#clang++ -Ofast -DTEST data_loader.cpp -o data_loader
clang++ -shared -fPIC -Ofast data_loader.cpp -o data_loader.so
