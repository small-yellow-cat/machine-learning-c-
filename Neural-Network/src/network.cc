#include "../include/network.hpp"
#include "../include/layer.hpp"
#include "../../include/DataHandler.h"
#include <numeric>
#include <algorithm>

//spec=the number of neurons for each traininglayer and inputSize=size of feature vector
//numclasses the size of output layer
Network::Network(std::vector<int> spec, int inputSize, int numClasses, double learningRate)
{
    for(int i = 0; i < spec.size(); i++)
    {
        if(i == 0)
            layers.push_back(new Layer(inputSize, spec.at(i)));
        else
            layers.push_back(new Layer(layers.at(i-1)->neurons.size(), spec.at(i)));
        
    }
    layers.push_back(new Layer(layers.at(layers.size()-1)->neurons.size(), numClasses));
    this->learningRate = learningRate;
}

Network::~Network() {}

//dot product of weight(1st) and input vector Ax+b
double Network::activate(std::vector<double> weights, std::vector<double> input)
{
    double activation = weights.back(); // bias term the last term
    for(int i = 0; i < weights.size() - 1; i++)
    {
        activation += weights[i] * input[i];
    }
    return activation;
}

//activation function
double Network::transfer(double activation)
{
    return 1.0 / (1.0 + exp(-activation));
}

//the derivative of activation function
double Network::transferDerivative(double output)
{
    return output * (1 - output);
}

std::vector<double> Network::fprop(Data *data)
{
    std::vector<double> inputs = *data->getNormalizedFeatureVector();
    for(int i = 0; i < layers.size(); i++)
    {
        Layer *layer = layers.at(i);
        std::vector<double> newInputs; //newInputs to overide the inputs after 1 layer
        for(Neuron *n : layer->neurons)
        {
            double activation = this->activate(n->weights, inputs);
            n->output = this->transfer(activation);
            //AX+b X:input  for A: row 1 neuron 1 row 2 neuron 2
            newInputs.push_back(n->output);
        }
        inputs = newInputs;
    }
    return inputs; // output layer outputs
}

//calculating the error contribution from each layer by the gradient
void Network::bprop(Data *data)
{
    for(int i = layers.size() - 1; i >= 0; i--)
    {
        Layer *layer = layers.at(i);
        std::vector<double> errors; //the gradient of errors devided by neurons' output in this layer
        if(i != layers.size() - 1)
        {
            for(int j = 0; j < layer->neurons.size(); j++)
            {   //the gradient of errors devided by neurons' output in this layer 
                 //from the contribution from beyond layers
                double error = 0.0;
                for(Neuron *n : layers.at(i + 1)->neurons)
                {
                    error += (n->weights.at(j) * n->delta);
                }
                errors.push_back(error);
            }
        } else {
            for(int j = 0; j < layer->neurons.size(); j++)
            {    //the term of the last layer is the expect-actual
                Neuron *n = layer->neurons.at(j);
                errors.push_back((double)data->getClassVector().at(j) - n->output); // expected - actual
            }
        }
        //the delta(gradient) for each neuron in current layer
        for(int j = 0; j < layer->neurons.size(); j++)
        {
            Neuron *n = layer->neurons.at(j);
            //delta =the gradient of error devided by the ouput of neuron in this layer
            //multiplied by derivative of activation function in this layer
            n->delta = errors.at(j) * this->transferDerivative(n->output); //gradient / derivative part of back prop.
        }
    }
}

void Network::updateWeights(Data *data)
{   
    std::vector<double> inputs = *data->getNormalizedFeatureVector();
    for(int i = 0; i < layers.size(); i++)
    {   //update the weight of neurons in one layer
        if(i != 0)
        {
            for(Neuron *n : layers.at(i - 1)->neurons)
            //inputs from neurons in the previous layer
            {
                inputs.push_back(n->output);
            }
        }
        for(Neuron *n : layers.at(i)->neurons)
        {  //updates the weight of neurons in the current layer by the err
            for(int j = 0; j < inputs.size(); j++)
            {
                n->weights.at(j) += this->learningRate * n->delta * inputs.at(j);
            }
            n->weights.back() += this->learningRate * n->delta; //update the basis
        }
        inputs.clear();
    }
}

int Network::predict(Data * data)
{
    std::vector<double> outputs = fprop(data);
    //return the index of max_elements in class vector
    return std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
}

void Network::train(int numEpochs)
{     //numEpochs=num of iteration
    for(int i = 0; i < numEpochs; i++)
    {   //the error of all the data in one training set
        double sumError = 0.0;
        for(Data *data : *this->trainingData)
        {   //using each data in training set
            std::vector<double> outputs = fprop(data);
            std::vector<int> expected = data->getClassVector();
            double tempErrorSum = 0.0;  //the error for one data
            for(int j = 0; j < outputs.size(); j++)
            {
                tempErrorSum += pow((double) expected.at(j) - outputs.at(j), 2);
            }
            sumError += tempErrorSum;
            bprop(data);
            //update the weights the defualt batch size is only one
            updateWeights(data);
        }
        printf("Iteration: %d \t Error=%.4f\n", i, sumError);
    }
}

double Network::test()
{
    double numCorrect = 0.0;
    double count = 0.0;
    for(Data *data : *this->testData)
    {
        count++;
        int index = predict(data);
        if(data->getClassVector().at(index) == 1) numCorrect++;
    }

    testPerformance = (numCorrect / count);
    return testPerformance;
}

void Network::validate()
{
    double numCorrect = 0.0;
    double count = 0.0;
    for(Data *data : *this->validationData)
    {
        count++;
        int index = predict(data);
        if(data->getClassVector().at(index) == 1) numCorrect++;
    }
    printf("Validation Performance: %.4f\n", numCorrect / count);
}

int main()
{
    DataHandler *dataHandler = new DataHandler();
#ifdef MNIST
    dataHandler->readInputData("../data/train-images.idx3-ubyte");
    dataHandler->readLabelData("../data/train-labels.idx1-ubyte");
    dataHandler->countClasses();
    dataHandler->normalize();
#else
    dataHandler->readCsv("../data/iris.data", ",");
#endif
    dataHandler->splitData();
    std::vector<int> hiddenLayers = {10};
    auto lambda = [&]() {
        Network *net = new Network(
            hiddenLayers, 
            dataHandler->getTrainingData()->at(0)->getNormalizedFeatureVector()->size(), 
            dataHandler->getClassCounts(),
            0.25);
        net->setTrainingData(dataHandler->getTrainingData());
        net->setTestData(dataHandler->getTestData());
        net->setValidationData(dataHandler->getValidationData());
        net->train(15);
        //the validation is the same as test here
        net->validate();
        printf("Test Performance: %.3f\n", net->test());
    };
    lambda();
}