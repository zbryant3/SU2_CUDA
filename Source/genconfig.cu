/**
 * Author: Zachariah Bryant
 * Description: Pre-generates a thermalized lattice configuration for later use.
 */

 //  ********************
 //  *      Headers     *
 //  ********************

#include <sys/stat.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include "./Headers/Complex.cuh"
#include "./Headers/LattiCuda.cuh"

using namespace std;

//  ************************************
//  *      Definition of Variables     *
//  ************************************
#define LATTSIZE 16
#define BETA 5.7
#define THERMAL 1000

//  **************************
//  *      Main Function     *
//  **************************
int main()
{
        LattiCuda model(LATTSIZE, BETA);

        for(int i = 0; i < THERMAL; i++){
          model.equilibrate();
        }
        model.save();

        return 0;
}
