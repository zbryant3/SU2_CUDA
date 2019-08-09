

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "./Headers/Complex.cuh"

using namespace std;


__device__ void MaMult(bach::complex<double> *m1, bach::complex<double> *m2, bach::complex<double> *r){

  bach::complex<double> neg = bach::complex<double> (-1,-1);

        r[0] = m1[0]*m2[0] + m1[1]*m2[2];
        r[1] = m1[0]*m2[1] + m1[1]*m2[3];
        r[2] = neg*bach::conj(r[1]);
        r[3] = bach::conj(r[0]);

};

__global__ void func(){
  bach::complex<double> m (1,9);
  bach::complex<double> n (2, 4);
  m = m / 6 ;
  bach::print(m);
}


int main(){

        func<<<1,1>>>();
        cudaDeviceSynchronize();


        return 0;
}
