//*****************************************************
// Usage: Generates the average of two polykov loops  *
//  across the lattice space.                         *
//                                                    *
// Author: Zachariah Bryant                           *
// Co-Author: Sebastian Dawid                         *
//*****************************************************


//**************
//   Headers   *
//**************
#include <sys/stat.h> //For checking file existance
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>

//Contains class wrap for SU model to be performed on the gpu
#include "./Headers/LattiCuda.cuh"



using namespace std;



//**************************************
//   Definition of all the variables   *
//**************************************
#define LATTSIZE 16
#define BETA 5.7
#define CONFIGS 10
#define THERMAL 1000
#define SEPARATION 100

//*****************************
//    Function Definitions    *
//*****************************

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
          if(*iter == 1){
            for(int i = 0; i < 4; i++) {
              name.pop_back();
            }

          }
          else{
            for(int i = 0; i < 5; i++){
              name.pop_back();
            }
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

        fstream file;

        file.open(polyname(), ios::out | ios:: trunc);

        //Thermalize the lattice
        for(int i = 0; i < THERMAL; i++){
          cout << i << endl;
          model.Equilibrate();
        }

        model.Save();

        //Generate a given amount of configs
        for(int i = 0; i < CONFIGS; i++){

          //Equilibrate to separate measurements
          for(int e = 0; e < SEPARATION; e++){
            cout << e << endl;
            model.Equilibrate();
          }

          //Gather the average of two polykov loops for different
          //distances for the config
          for(int dist = 1; dist < 14; dist++){
            file << model.Polykov(dist) << " ";
          }
          file << "\n";
          file.flush();

        }

        file.close();

        return 0;
}
