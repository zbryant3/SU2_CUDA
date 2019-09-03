/**
 * Author: Zachariah Bryant
 * Description: This program times the average equilibration of 100 steps
 *              vs the lattice size.
 */


//  ********************
//  *      Headers     *
//  ********************
#include <iostream>
#include <fstream>
#include <string>
#include <chrono> //< For Timer
#include "./Headers/LattiCuda.cuh"

using namespace std;
using namespace std::chrono;

//  ***********************************
//  *     Definition of Variables     *
//  ***********************************
#define LATTSIZE 8
#define BETA 2.8

//  **************************
//  *      Main Function     *
//  **************************
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
                        model.equilibrate();
                }

                high_resolution_clock::time_point t2 = high_resolution_clock::now();
                avg = duration_cast<milliseconds>( t2 - t1 ).count();

                File << avg/100 << " " << latt << "\n";
                File.flush();

        }

        File.close();


        return 0;
}
