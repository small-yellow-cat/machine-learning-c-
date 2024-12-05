#!/bin/bash
#mkdir obj
#mkdir lib
g++  -std=c++11  -o ./obj/data.o -I./include -c ./src/data.cc
g++  -std=c++11  -o ./obj/DataHandler.o -I./include -c ./src/DataHandler.cc
g++  -std=c++11  -o ./obj/common.o -I./include -c ./src/common.cc
ar rcs ./lib/libdata.a ./obj/*.o
