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
#define LATTSIZE 40
#define BETA 2




using namespace std;


//**********************
//    Main Function    *
//**********************
int main()
{
        LattiCuda model(LATTSIZE, BETA);

        fstream File;
        double temp;

        File.open("../Data/AvgPlaq_vs_Equilibration.dat", ios::out | ios::trunc);

        double avg{0};
        for(int i = 0; i < 2; i++){
          temp = model.AvgPlaquette();
          avg += temp;
          cout << "\nAvgPlaquette:\t" << temp << "\n";
          File << i << " " << temp << "\n";
          File.flush();
          model.Equilibrate();
        }
        File.close();
        cout << "Average: " << avg/200 <<"\n";

        return 0;
}
