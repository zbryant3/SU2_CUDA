/**
 * Author: Zachaiah Bryant
 * Description: Generates the average of two polykov loops across
 *              distances 1-16 for various SU(2) lattice configurations.
 */


//  *******************
//  *      Headers    *
//  *******************
#include <sys/stat.h> //For checking file existance
#include <iostream>
#include <fstream>
#include <string>
#include "./Headers/LattiCuda.cuh"

using namespace std;

//  *************************************
//  *      Definition of Variables      *
//  *************************************

#define LATTSIZE 16
#define BETA 2.5
#define CONFIGS 10
#define SEPARATION 100

//  ***************************
//  *      Function Headers   *
//  ***************************

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
                 if(*iter == 1) {
                         for(int i = 0; i < 4; i++) {
                                 name.pop_back();
                         }
                 }
                 else if(*iter <= 10) {
                         for(int i = 0; i <= 4; i++) {
                                 name.pop_back();
                         }
                 }
                 else if(*iter <= 100) {
                         for(int i = 0; i <= 5; i++) {
                                 name.pop_back();
                         }
                 }
                 else{
                         for(int i = 0; i <= 6; i++) {
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


 //  **************************
 //  *      Main Function     *
 //  **************************
int main()
{

        LattiCuda model(LATTSIZE, BETA);

        fstream file;
        file.open(polyname(), ios::out | ios::trunc);

        //Open Pre-Thermalized Lattice
        model.load();

        //Equilibrate the loaded lattice a couple times
        for(int i = 0; i < 10; i++){
          model.equilibrate();
        }

        //Generate a given amount of configs
        for(int i = 0; i < CONFIGS; i++) {
                cout << i << endl;

                //Equilibrate to separate measurements
                for(int e = 0; e < SEPARATION; e++) {
                        model.equilibrate();
                }

                //Gather the average of two polykov loops for different
                //distances for the config
                for(int dist = 1; dist <= 16; dist++) {
                        file << -log(model.polykov(dist))/LATTSIZE << " ";
                }
                file << "\n";
                file.flush();

        }

        file.close();

        return 0;
}
