#include "../include/neuron.hpp"
#include <random>

double generateRandomNumber(double min, double max)
{
    double random = (double) rand() / RAND_MAX;
    return min + random * (max - min);
}

Neuron::Neuron(int previousLayerSize, int currentLayerSize)
{
    initializeWeights(previousLayerSize);
}

//for each neuron scalar output AX+b
void Neuron::initializeWeights(int previousLayerSize)
{   
    std::default_random_engine generator;
    const double epsilon = 1e-8;
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    //std::normal_distribution<double> distribution(-1.0, 1.0);
    for(int i = 0; i < previousLayerSize + 1; i++)
    {   double weig=distribution(generator);
        if (weig == 0.0) { weig = epsilon; 
        } 
        else if (weig > -epsilon && weig < epsilon) { weig = (weig > 0) ? epsilon : -epsilon; }
        weights.push_back(weig);
        //weights.push_back(generateRandomNumber(-1.0, 1.0));
    }
}