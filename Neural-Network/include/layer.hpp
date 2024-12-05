#ifndef __LAYER_HPP
#define __LAYER_HPP
#include "neuron.hpp"
#include <stdint.h>
#include <vector>

class Layer {

  public:
    int currentLayerSize; //number of neuron
    std::vector<Neuron *> neurons; //the neurons which forms the layer
    std::vector<double> layerOutputs;
    Layer(int, int); //previous layer size and current layer size
};
#endif
