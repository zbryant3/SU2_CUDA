#include "./Headers/LattiCuda.cuh"
#include "./Headers/LattiCuda_Device.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>


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
GPU_Initialize(thrust::complex<double> *d_lattice, int tdim){

        LattiCuda_Device device(&d_size, &d_beta, d_lattice, NULL, tdim);

        device.Initialize();
};


/**
 * Equilibrates the lattice using the GPU.
 * @param  d_lattice - Pointer to the lattice in device memory
 */
__global__ void
GPU_Equilibrate(thrust::complex<double> *d_lattice, int tdim, int dir){

        //Shared sublattice memory with size determined at kernal launch
        extern __shared__ thrust::complex<double> sub_lattice[];

        LattiCuda_Device device(&d_size, &d_beta, d_lattice, sub_lattice, tdim);

        device.Equilibrate(dir);

};



/**
 * Gets the average plaquette of the lattice
 */
__global__ void
GPU_AvgPlaquette(thrust::complex<double> *d_lattice, int tdim, double *d_plaq, double *d_iter){

        //Shared sublattice memory with size determined at kernal launch
        extern __shared__ thrust::complex<double> sub_lattice[];

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
        h_lattice = new thrust::complex<double>[memsize];

        //Create Device Lattice
        cudaMalloc((void**)&d_lattice, memsize*sizeof(thrust::complex<double>));

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

        int half = h_size/2;


        //Array to hold total avg plaquett per thread and total amount of iterations
        double h_plaq[h_size*h_size*h_size*h_size];
        double h_iter[h_size*h_size*h_size*h_size];
        double *d_plaq;
        double *d_iter;

        cudaMalloc((void**)&d_plaq, sizeof(double)*h_size*h_size*h_size*h_size);
        cudaMalloc((void**)&d_iter, sizeof(double)*h_size*h_size*h_size*h_size);

        //Dimensions for the kernal
        dim3 Threads(2, 2, 2);
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
