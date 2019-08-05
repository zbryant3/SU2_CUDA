#include "./Headers/LattiCuda.cuh"
#include "./Headers/LattiCuda_Device.cuh"
#include "./Headers/Complex.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>


using namespace std;

//*************************************
//    Global Variable Declarations    *
//*************************************

/**
 * Constant Variables for device code
 * @var   d_size    - Size of the lattice
 * @var   d_beta    - Value of Beta
 */
__constant__ int d_size;
__constant__ double d_beta;


//*********************************
//      GPU Kernal Functions      *
//*********************************

/**
 * Initializes all the links on the lattice to the unit matrix
 */
__global__ void
GPU_Initialize(bach::complex<double> *d_lattice, int tdim){

        LattiCuda_Device device(&d_size, &d_beta, d_lattice, NULL, tdim);

        device.Initialize();
};


/**
 * Equilibrates the lattice using the GPU.
 * @param  d_lattice - Pointer to the lattice in device memory
 */
__global__ void
GPU_Equilibrate(bach::complex<double> *d_lattice, int tdim, int dir){

        //Shared sublattice memory with size determined at kernal launch
        extern __shared__ bach::complex<double> sub_lattice[];

        LattiCuda_Device device(&d_size, &d_beta, d_lattice, sub_lattice, tdim);

        device.Equilibrate(dir);

};



/**
 * Gets the average plaquette of the lattice
 */
__global__ void
GPU_AvgPlaquette(bach::complex<double> *d_lattice, int tdim, double *d_plaq, double *d_iter){

        //Shared sublattice memory with size determined at kernal launch
        extern __shared__ bach::complex<double> sub_lattice[];

        LattiCuda_Device device(&d_size, &d_beta, d_lattice, sub_lattice, tdim);

        device.AvgPlaquette(d_plaq, d_iter);

};


//*******************************
//    Private Member Functions  *
//*******************************

/**
 * Initializes all the links on the lattice to the unit matrix
 */
__host__ void
LattiCuda::Initialize(){

        int half = h_size/2;
        dim3 in_Threads(2, 2, 2);
        dim3 in_Blocks(half, half, half);

        for(int t = 0; t < h_size; t++) {
                GPU_Initialize<<<in_Blocks,in_Threads>>>(d_lattice, t);
        }
};



/**
 * Returns 1D array location for linearized 4D lattice
 * @param  dim - Array with lattice dimension location t,x,y,z
 * @param  d   - Direction to look in
 * @param  m   - Matrix element for link
 * @return     - Int for array location
 */
__host__ int
LattiCuda::Loc(int *dim, int d, int m){
  int coor{0};

  coor = dim[1] + dim[2]*(h_size) + dim[3]*(h_size)*(h_size) + dim[0]*(h_size)*(h_size)*(h_size)
         + d*(h_size)*(h_size)*(h_size)*(h_size) + m*(h_size)*(h_size)*(h_size)*(h_size)*(4);

  return coor;
};



//*******************************
//    Public Member Functions   *
//*******************************

/**
 * Constructor for the Lattice QCD wrapper
 * @param   LattSize  - Size of desired lattice
 * @param   inBeta    - Beta value
 */
__host__
LattiCuda::LattiCuda(int LattSize, double inBeta){

        //Construct Host Variables
        h_size = LattSize;
        h_beta = inBeta;
        memsize = h_size*h_size*h_size*h_size*4*4;

        //Create Host Lattice
        h_lattice = new bach::complex<double>[memsize];

        //Create Device Lattice
        cudaMalloc((void**)&d_lattice, memsize*sizeof(bach::complex<double>));

        //Construct Constant Device Variables
        cudaMemcpyToSymbol(d_size, &h_size, sizeof(int));
        cudaMemcpyToSymbol(d_beta, &h_beta, sizeof(double));

        //Initialize the lattice on creation
        Initialize();

};


/**
 * Destructor for the Lattice QCD wrapper
 */
__host__
LattiCuda::~LattiCuda(){

        delete[] h_lattice;

        cudaFree(d_lattice);

};


/**
 * Equilibrates the lattice
 */
__host__ void
LattiCuda::Equilibrate(){

        int half = h_size/4;

        //Dimensions for the kernal
        dim3 Threads(4, 4, 4);
        dim3 Blocks(half, half, half);


        int sharedsize = 0;
/*
        //Max shared size is 49152
        int sharedsize = ((h_size)/(half) + 2) * ((h_size)/(half) + 2)
 * ((h_size)/(half) + 2) * 768;

        //Ensures shared size isnt too large
        if(sharedsize > 49152) {
                cout << "Shared memory size too large. Exiting... \n \n";
                exit(EXIT_FAILURE);
        }
 */

        //All directions need to updated independently
        for(int d = 0; d < 4; d++) {

                //Checkerboard pattern for T dimension
                for(int offset = 0; offset <= 1; offset++) {
                        for(int tdim = 0; tdim < half; tdim++) {
                                GPU_Equilibrate<<<Blocks, Threads, sharedsize>>>(d_lattice, ((tdim)*2 + offset), d);
                        }
                        cudaDeviceSynchronize();
                }
        }

};



/**
 * Gets the value of the average plaquette of the lattice
 * @return double - Average Plaquette
 */
__host__ double
LattiCuda::AvgPlaquette(){

        int half = h_size/4;
        //Dimensions for the kernal
        dim3 Threads(4, 4, 4);
        dim3 Blocks(half, half, half);


        //Array to hold total avg plaquett per thread and total amount of iterations
        double h_plaq[h_size*h_size*h_size*h_size];
        double h_iter[h_size*h_size*h_size*h_size];
        double *d_plaq;
        double *d_iter;

        cudaMalloc((void**)&d_plaq, sizeof(double)*h_size*h_size*h_size*h_size);
        cudaMalloc((void**)&d_iter, sizeof(double)*h_size*h_size*h_size*h_size);




        int sharedsize = 0;
        /*
           //Max shared size is 49152
           int sharedsize = ((h_size)/(half) + 2) * ((h_size)/(half) + 2)
         * ((h_size)/(half) + 2) * 768;
           //Ensures shared size isnt too large
           if(sharedsize > 49152) {
                cout << "Shared memory size too large. Exiting... \n \n";
                exit(EXIT_FAILURE);
           }
         */


        //Run on gpu for each time slice
        for(int tdim = 0; tdim < h_size; tdim++) {
                GPU_AvgPlaquette<<<Blocks, Threads, sharedsize>>>
                (d_lattice, tdim, d_plaq, d_iter);
        }
        cudaDeviceSynchronize();

        //Copy results from gpu
        cudaMemcpy(h_plaq, d_plaq, sizeof(double)*h_size*h_size*h_size*h_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_iter, d_iter, sizeof(double)*h_size*h_size*h_size*h_size, cudaMemcpyDeviceToHost);


        //Evaluate results
        double totplaq{0};
        double totiter{0};
        for(int i = 0; i < h_size*h_size*h_size*h_size; i++) {
                totplaq += h_plaq[i];
                totiter += h_iter[i];
        }

        cudaFree(d_plaq);
        cudaFree(d_iter);

        return (1 - totplaq/totiter);
};





/**
 * Saves the lattice configuration to a file.
 */
__host__ void
LattiCuda::Save(){
  printf("Saving Lattice Configuration......\n");

  //Copy device lattice to host lattice
  cudaMemcpy(h_lattice, d_lattice, memsize*sizeof(bach::complex<double>), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  //File to write to
  fstream File1;
  File1.open("../Data/LatticeConfig.dat", ios::out | ios::trunc);

  int pos[4] = {0,0,0,0};
  for( pos[0] = 0; pos[0] < h_size; pos[0]++){ // T dimension
    for( pos[1] = 0; pos[1] < h_size; pos[1]++){ // X dimension
      for( pos[2] = 0; pos[2] < h_size; pos[2]++){ // Y dimension
        for( pos[3] = 0; pos[3] < h_size; pos[3]++){ // Z dimension
          for(int d = 0; d < 4; d++){ // direction
            File1 << h_lattice[Loc(pos, d, 0)].real() << " " << h_lattice[Loc(pos, d, 0)].imag()
            << " " << h_lattice[Loc(pos, d, 1)].real() << " " << h_lattice[Loc(pos, d, 1)].imag() << endl;
          }
        }
      }
    }
  }

  File1.close();

  printf("Done Saving.\n");


};



/**
 * Loads a lattice configuration from a file
 * @param  file - File to load from
 */
__host__ void
LattiCuda::Load(string file){
  printf("Loading Lattice Configuration.......\n");

  fstream File;
  File.open(file, ios::in);

  double real, imag;


  int pos[4] = {0,0,0,0};
  for( pos[0] = 0; pos[0] < h_size; pos[0]++){ // T dimension
    for( pos[1] = 0; pos[1] < h_size; pos[1]++){ // X dimension
      for( pos[2] = 0; pos[2] < h_size; pos[2]++){ // Y dimension
        for( pos[3] = 0; pos[3] < h_size; pos[3]++){ // Z dimension
          for(int d = 0; d < 4; d++){ // direction
            File >> real;
            File >> imag;
            h_lattice[Loc(pos, d, 0)] = bach::complex<double>(real, imag);
            h_lattice[Loc(pos, d, 3)] = bach::complex<double>(real, (-1)*imag);

            File >> real;
            File >> imag;
            h_lattice[Loc(pos, d, 1)] = bach::complex<double>(real, imag);
            h_lattice[Loc(pos, d, 2)] = bach::complex<double>((-1)*real, imag);
          }
        }
      }
    }
  }

  File.close();

  //Copy host lattice to device lattice
  cudaMemcpy(d_lattice, h_lattice, memsize*sizeof(bach::complex<double>), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  printf("Done Loading.\n");


};
