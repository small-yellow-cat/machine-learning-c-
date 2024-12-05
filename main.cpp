#include "include/data.h"
#include "include/common.hpp"
#include "include/DataHandler.h"

int main(){
DataHandler *dh=new DataHandler();
dh->readInputData("data/train-images.idx3-ubyte");
dh->readLabelData("data/train-labels.idx1-ubyte");
dh->splitData();
dh->countClasses();


}