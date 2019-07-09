//*****************************************************
// Usage: Performs SU[2] simulations utilizing        *
//  monte carlo calculations performed on the GPU.    *
//                                                    *
// Author: Zachariah Bryant                           *
//*****************************************************


//**************
//   Headers   *
//**************
#include <iostream>
#include <fstream>
#include <string.h>
#include <vector>

//Contains class wrap for SU model to be performed on the gpu
#include "./Headers/LattiCuda.cuh"


//**************************************
//   Definition of all the variables   *
//**************************************
#define LATTSIZE 16
#define SIZEMATRIX 2




using namespace std;


//**********************
//    Main Function    *
//**********************
int main()
{
  LattiCuda model(LATTSIZE);

  model.Equilibrate();


  return 0;
}
