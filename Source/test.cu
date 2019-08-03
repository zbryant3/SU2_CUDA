

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "./Headers/Complex.cuh"

using namespace std;

__global__ void foo(){
  bach::complex<double> c(66,33);
  bach::complex<double> b(1,2);
  bach::complex<double> a(0,0);
  a = b + c;
  bach::print(a);

};


int main(){
    foo<<<1,1>>>();
    cudaDeviceSynchronize();


    return 0;
}
