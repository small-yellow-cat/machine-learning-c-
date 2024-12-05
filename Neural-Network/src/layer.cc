#include "layer.hpp"

//1st(2nd) parameter represents the number of neurons in previous(current) layer
Layer::Layer(int previousLayerSize, int currentLayerSize)
{
    for(int i = 0; i < currentLayerSize; i++)
    {
        neurons.push_back(new Neuron(previousLayerSize, currentLayerSize));
    }
    this->currentLayerSize = currentLayerSize;
}