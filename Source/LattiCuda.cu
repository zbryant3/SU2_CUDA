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
 * @var         [description]
 */
__constant__ int d_size;


//*********************************
//      GPU Kernal Functions      *
//*********************************

__global__ void
GPU_test(thrust::complex<double> *d_lattice){

        LattiCuda_Device Testing(&d_size, d_lattice);

        Testing.TestBack();

};

/**
 * Initializes all the links on the lattice to the unit matrix
 */
__global__ void
GPU_Initialize(thrust::complex<double> *d_lattice){

        LattiCuda_Device device(&d_size, d_lattice);

        device.Initialize();
};


/**
 * Equilibrates the lattice using the GPU.
 * @param  d_lattice - Pointer to the lattice in device memory
 */
__global__ void
GPU_Equilibrate(thrust::complex<double> *d_lattice, int tdim){

        //Shared sublattice memory with size determined at kernal launch
        extern __shared__ thrust::complex<double> sub_lattice[];

        LattiCuda_Device device(&d_size, d_lattice, sub_lattice, tdim);

        device.Equilibrate();

};


//*******************************
//    Private Member Functions  *
//*******************************

/**
 * Initializes all the links on the lattice to the unit matrix
 */
__host__ void
LattiCuda::Initialize(){

        //                X           Y split            Z split
        dim3 in_Threads(h_size, h_size/(h_size/2), h_size/(h_size/2));

        //  sizeofsplit:   Y           Z         T-Dimension
        dim3 in_Blocks((h_size/2), (h_size/2), h_size);

        GPU_Initialize<<<in_Blocks,in_Threads>>>(d_lattice);
};





//*******************************
//    Public Member Functions   *
//*******************************

/**
 * Constructor for the Lattice QCD wrapper
 */
__host__
LattiCuda::LattiCuda(int LattSize){

        //Construct Host Variables
        h_size = LattSize;
        memsize = h_size*h_size*h_size*h_size*4*4;

        //Create Host Lattice
        h_lattice = new thrust::complex<double>[memsize];

        //Create Device Lattice
        cudaMalloc((void**)&d_lattice, memsize*sizeof(thrust::complex<double>));

        //Initialize the lattice on creation
        Initialize();

        //Construct Constant Device Variables
        cudaMemcpyToSymbol(d_size, &h_size, sizeof(int));

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

        int half = h_size/2;

        //Dimensions for the kernal
        dim3 Threads(h_size/half, h_size/half, h_size/half);
        dim3 Blocks(half, half, half);

        //Max shared size is 49152
        int sharedsize = ((h_size)/(half) + 2) * ((h_size)/(half) + 2)
                         * ((h_size)/(half) + 2) * 768;

        //Ensures shared size isnt too large
        if(sharedsize > 49152){
          cout << "Shared memory size too large. Exiting... \n \n";
          exit(EXIT_FAILURE);
        }

        //Checkerboard pattern for T dimension
        for(int offset = 0; offset <= 1; offset++){
          for(int tdim = 0; tdim < half; tdim++){
            GPU_Equilibrate<<<Blocks, Threads, sharedsize>>>(d_lattice, (tdim)*2 + offset);
          }
          cudaDeviceSynchronize();
        }

};



/**
 * Initiates a test for various reasons
 */
__host__ void
LattiCuda::Test(){

        //double arrsize = memsize*sizeof(thrust::complex<double>);

        //cout << "Size of Arrary " << arrsize << " bytes\n";
        //cout << "Number of Blocks Required ~ " << arrsize/49152 << " \n";

        //GPU_test<<<10,10>>>(d_lattice);

};
