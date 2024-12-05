#ifndef __DATA_HANDLER_H
#define __DATA_HANDLER_H

#include "fstream"
#include "stdint.h"
#include "data.h"
#include <vector>
#include <string>
#include <map>
#include <unordered_set>
#include <math.h>

class DataHandler
{
  std::vector<Data *> *dataArray; // all of the data
  std::vector<Data *> *trainingData;
  std::vector<Data *> *testData;
  std::vector<Data *> *validationData;
  int class_counts;
  int featureVectorSize;
  std::map<uint8_t, int> classFromInt;     //uint8_t type label -> 
  std::map<std::string, int> classFromString; //string type label->

  public:
  const double TRAIN_SET_PERCENT = .1;
  const double TEST_SET_PERCENT = .075;
  const double VALID_SET_PERCENT = 0.005;
  //construction and destruction function

  DataHandler();
  ~DataHandler();
  
  //reading file, feature vector label
  void readCsv(std::string, std::string);
  void readInputData(std::string path);
  void readLabelData(std::string path);

 //handle and print the data
  void splitData();
  void countClasses();
  void normalize();
  void print();

  //get the size  and number of class
  int getClassCounts();
  int getDataArraySize();
  int getTrainingDataSize();
  int getTestDataSize();
  int getValidationSize();

  //
  uint32_t format(const unsigned char* bytes);
  
  //get the data
  std::vector<Data *> * getTrainingData();
  std::vector<Data *> * getTestData();
  std::vector<Data *> * getValidationData();
  std::map<uint8_t, int> getClassMap();

};

#endif
