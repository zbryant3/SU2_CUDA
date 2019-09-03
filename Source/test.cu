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

using namespace std;

//  **********************************
//  *      Definition of Variables   *
//  **********************************
#define LATTSIZE 16
#define BETA 5.7

//  ***************************
//  *     Extra Functions     *
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
        fstream file;

        for(int i = 0; i < 10; i++) {
                file.open(polyname(), ios::out | ios::trunc);
                file << i << "\n";
                file.flush();
                file.close();
        }


        return 0;
}
