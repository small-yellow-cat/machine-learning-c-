#!/bin/bash
#mkdir obj
g++ -c -std=c++11 -g -DMNIST  src/neuron.cc -o obj/neuron.o -I./include -I../include
g++ -c -std=c++11 -g -DMNIST src/layer.cc -o obj/layer.o -I./include -I../include
g++  -std=c++11 -g  -DMNIST src/network.cc src/neuron.cc src/layer.cc -o main2.exe -I./include -I../include -L../lib -ldata 
