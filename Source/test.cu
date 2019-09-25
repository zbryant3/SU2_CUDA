/**
 * Author: Zachariah Bryant
 * Description: Testing file for various purposes.
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
#include <vector>

using namespace std;

//  **********************************
//  *      Definition of Variables   *
//  **********************************
#define FILES 15

//  ***************************
//  *     Extra Functions     *
//  ***************************


double average(double *avgspin)
{
  double total{0};

  for(unsigned int i = 0; i < FILES*10; i++)
  total += avgspin[i];

  return total/(FILES*10);
}

double standard_Deviation(double *avgspin)
{
  double x{0};
  double y{average(avgspin)};

  for(int i = 0; i < FILES*10; i++)
  x += pow((avgspin[i] - y), 2);

  x = sqrt(x)/sqrt(FILES*10*(FILES*10-1));
  return x;
}



/**
 * Checks for the existance of a file
 * @param  name - Destination and name to look for
 */
inline bool exist(const std::string& name) {
        struct stat buffer;
        return (stat (name.c_str(), &buffer) == 0);
}


/**
 * Function for creating a unique file for the polykov loop
 * @return string of name and file location
 */
std::string polyname(int i) {
        string name = "../Data/Polykov/PolyVsDist";
        name += std::to_string(i);
        name += ".dat";

        return name;
};


//  **************************
//  *      Main Function     *
//  **************************
int main()
{
        fstream fileout, filein;
        double data[16][FILES*10];

        fileout.open("../Data/DistvsPolykov.dat", ios::out | ios:: trunc);

        for(int i = 0; i <= FILES; i++) {
                filein.open(polyname(i), ios::in);

                for(int dis = 0; dis < 16; dis++){
                  for(int c = 0; c < 10; c++){

                    filein >> data[dis][i*c];
                  }
                }

                filein.close();
        }

        for(int i = 0; i < 16; i++){
          fileout << i + 1<< " "<< average(data[i]) << " " << standard_Deviation(data[i]) << endl;
          //fileout << i + 1 << " " <<  (-1*log(average(data[i])))/(FILES*10) << " " << (-1*log(standard_Deviation(data[i])))/(FILES*10) << "\n";
          fileout.flush();
        }

        fileout.close();


        return 0;
}
