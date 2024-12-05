#include "../include/data.h"

//...................
//set feature vector or normalized feature vector
void Data::setFeatureVector(std::vector<uint8_t>* vect)
{
  featureVector = vect;
}

void Data::setNormalizedFeatureVector(std::vector<double>* vect)
{
  normalizedFeatureVector = vect;
}

//.............
//append to the feature and normalized feature vector
void Data::appendToFeatureVector(uint8_t val)
{
  featureVector->push_back(val);
}

void Data::appendToFeatureVector(double val)
{
  normalizedFeatureVector->push_back(val);
}

//.................
//get feature vector and its size
std::vector<uint8_t> * Data::getFeatureVector()
{
  return featureVector;
}

std::vector<double> * Data::getNormalizedFeatureVector()
{
  return normalizedFeatureVector;
}

int Data::getFeatureVectorSize()
{
  return featureVector->size();
}

//............
//print feature vector

void Data::printVector()
{
  printf("[ ");
  for(uint8_t val : *featureVector)
  {
    printf("%u ", val);
  }
  printf("]\n");
}

void Data::printNormalizedVector()
{
  printf("[ ");
  for(auto val : *normalizedFeatureVector)
  {
    printf("%.2f ", val);
  }
  printf("]\n");
  
}

//.................
//set  class vector
void Data::setClassVector(int classCounts)
{
  classVector = new std::vector<int>();
  for(int i = 0; i < classCounts; i++)
  {
    if(i == label)
      classVector->push_back(1);
    else
      classVector->push_back(0);
  }
}

std::vector<int>  Data::getClassVector()
{
  return *classVector;
}

//........................
//label
void Data::setLabel(uint8_t val)
{
  label = val;
}

void Data::setEnumeratedLabel(uint8_t val)
{
  enumeratedLabel = val;
}

uint8_t Data::getLabel()
{
  return label;
}

uint8_t Data::getEnumeratedLabel()
{
  return enumeratedLabel;
}

//.............
//distance
void Data::setDistance(double dist)
{
  distance = dist;
}

double Data::getDistance()
{
  return distance;
}






