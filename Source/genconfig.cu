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
#include <chrono>

//Contains class wrap for SU model to be performed on the gpu
#include "./Headers/LattiCuda.cuh"


//**************************************
//   Definition of all the variables   *
//**************************************
#define LATTSIZE 32
#define BETA 2.8




using namespace std;


//**********************
//    Main Function    *
//**********************
int main()
{
        LattiCuda model(LATTSIZE, BETA);
        model.Load("../Data/LatticeConfig.dat");
        cout << "program does something\n";
        double temp = model.AvgPlaquette();
        cout << temp << "\n";
        model.Equilibrate();
        cudaDeviceSynchronize();



        return 0;
}
