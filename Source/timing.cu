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
#define LATTSIZE 8
#define BETA 2.8




using namespace std;
using namespace std::chrono;


//**********************
//    Main Function    *
//**********************
int main()
{

        fstream File;
        double avg;


        File.open("../Data/Time_vs_Size.dat", ios::out | ios::trunc);
        for(int i = 1; i <= 3; i++) {

                int latt = i*8;
                LattiCuda model(latt, BETA);

                high_resolution_clock::time_point t1 = high_resolution_clock::now();

                for(int i = 0; i < 100; i++) {
                        model.Equilibrate();
                }

                high_resolution_clock::time_point t2 = high_resolution_clock::now();
                avg = duration_cast<milliseconds>( t2 - t1 ).count();

                File << avg/100 << " " << latt << "\n";
                File.flush();

        }

        File.close();


        return 0;
}
