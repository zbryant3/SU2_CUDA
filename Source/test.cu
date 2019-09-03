
#include <sys/stat.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include "./Headers/Complex.cuh"
//Contains class wrap for SU model to be performed on the gpu
#include "./Headers/LattiCuda.cuh"


//**************************************
//   Definition of all the variables   *
//**************************************
#define LATTSIZE 16
#define BETA 5.7

using namespace std;


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
std::string polyname() {
  string name = "../Data/Polykov/PolyVsDist";
  name += ".dat";

  int *iter = new int;
  *iter = 0;
  while(exist(name)) {

          *iter += 1;

          //Gets rid of .dat
          for(int i = 0; i < 4; i++) {
                  name.pop_back();
          }

          name += std::to_string(*iter);
          name += ".dat";

  }
  delete iter;

  std::cout << name << "\n";

  return name;
};


//**********************
//    Main Function    *
//**********************
int main()
{
          LattiCuda model(LATTSIZE, BETA);

          for(int i = 0; i < 100; i++){
            cout << i << endl;
            model.Equilibrate();
          }

          model.Save();



        /*


           double temp;

           for(int i = 0; i < 2; i++){
           model.Equilibrate();
           temp = model.Polykov(1);
           cout << temp << "\n";
           }
         */

        return 0;
}
