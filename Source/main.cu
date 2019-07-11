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
#define LATTSIZE 8
#define BETA 2.7




using namespace std;


//**********************
//    Main Function    *
//**********************
int main()
{
        LattiCuda model(LATTSIZE, BETA);

        for(int i = 0; i < 1; i++){
          cout << model.AvgPlaquette() << "\n";
          model.Equilibrate();
        }


        return 0;
}
