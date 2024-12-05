#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "../../include/data.h"
#include "neuron.hpp"
#include "layer.hpp"
#include "../../include/common.hpp"

class Network : public CommonData
{
  public:
    std::vector<Layer *> layers;
    double learningRate;
    double testPerformance;
    //construction and destruction 
    Network(std::vector<int> spec, int, int, double);
    ~Network();
    std::vector<double> fprop(Data *data); //return the vector output of last layer
    //dot product of weight(1st) and input vector
    double activate(std::vector<double>, std::vector<double>); 
    //using activation function such as sigma
    double transfer(double);
    double transferDerivative(double); // used for backprop
    void bprop(Data *data);
    void updateWeights(Data *data);
    int predict(Data *data); // return the index of the maximum value in the output array.
    void train(int); // the parameter represents num iterations
    double test();
    void validate();
};

#endif
