#ifndef _NEURON_HPP_
#define _NEURON_HPP_
#include <stdio.h>
#include <vector>
#include <cmath>

class Neuron {
  public:
    double output;
//delta =the gradient of error devided by the ouput of neuron * derivative of corres activation function
    double delta; //needs to update the weight
    std::vector<double> weights;
    //constuction function
    Neuron(int, int);
    //weight
    void initializeWeights(int); 
};

#endif
