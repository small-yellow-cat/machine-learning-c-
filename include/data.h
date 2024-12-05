#ifndef __DATA_H
#define __DATA_H

#include <vector>
#include "stdint.h" // uint8_t 
#include "stdio.h"
class Data
{
  std::vector<uint8_t> *featureVector;
  std::vector<double> *normalizedFeatureVector;
  std::vector<int> *classVector;
  uint8_t label; 
  uint8_t enumeratedLabel; // label changed to number A -> 1
  double distance;

  public:
  //set feature vector or normalized feature vector
  void setFeatureVector(std::vector<uint8_t>*);
  void setNormalizedFeatureVector(std::vector<double>*);
  //append to feature vector
  void appendToFeatureVector(uint8_t);
  void appendToFeatureVector(double);
  //get feature vector and its size
  std::vector<uint8_t> * getFeatureVector();
  std::vector<double> * getNormalizedFeatureVector();
  int getFeatureVectorSize();
  //print feature vector
   void printVector();
  void printNormalizedVector();

  //set  class vector
  void setClassVector(int counts);
  std::vector<int> getClassVector();

  //label
  void setLabel(uint8_t);
  void setEnumeratedLabel(uint8_t);
  uint8_t getLabel();
  uint8_t getEnumeratedLabel();

  //distance
  void setDistance(double);
  double getDistance();

  

  

};

#endif
