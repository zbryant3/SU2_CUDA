/**
 * Author: Zachariah Bryant
 * Description: Generates the average plaquette of a SU(2) lattice
 *              vs equilibration steps for an unthermalized lattice.
 */


 //  ********************
 //  *      Headers     *
 //  ********************
#include <iostream>
#include <fstream>
#include <string.h>
#include "./Headers/LattiCuda.cuh"

using namespace std;

//  ************************************
//  *      Definition of Variables     *
//  ************************************
#define LATTSIZE 40
#define BETA 2

//  **************************
//  *      Main Function     *
//  **************************
int main()
{
        LattiCuda model(LATTSIZE, BETA);

        fstream File;
        double temp;

        File.open("../Data/AvgPlaq_vs_Equilibration.dat", ios::out | ios::trunc);

        double avg{0};
        for(int i = 0; i < 20; i++){
          temp = model.avgPlaquette();
          avg += temp;
          cout << "\nAvgPlaquette:\t" << temp << "\n";
          File << i << " " << temp << "\n";
          File.flush();
          model.equilibrate();
        }
        File.close();
        cout << "Average: " << avg/200 <<"\n";

        return 0;
}
