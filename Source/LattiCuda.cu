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

        LattiCuda_Device device(&d_size, &d_beta, d_lattice, tdim);

        device.Initialize();
};


/**
 * Equilibrates the lattice using the GPU.
 * @param  d_lattice - Pointer to the lattice in device memory
 */
__global__ void
GPU_Equilibrate(bach::complex<double> *d_lattice, int tdim, int dir){

        LattiCuda_Device device(&d_size, &d_beta, d_lattice, tdim);

        device.Equilibrate(dir);

};



/**
 * Gets the average plaquette of the lattice
 */
__global__ void
GPU_AvgPlaquette(bach::complex<double> *d_lattice, int tdim, double *d_plaq, double *d_iter){

        LattiCuda_Device device(&d_size, &d_beta, d_lattice, tdim);

        device.AvgPlaquette(d_plaq, d_iter);

};


/**
 * Gets the
 * @param d_lattice [description]
 * @param d_plaq    [description]
 * @param d_iter    [description]
 * @param dist      [description]
 */
__global__ void
GPU_Polykov(bach::complex<double> *d_lattice, double *d_poly, double *d_iter, int dist){

  //Create a gpu object with time slice set to zero
  LattiCuda_Device device(&d_size, &d_beta, d_lattice, 0);

  device.Polykov(d_poly, d_iter, dist);

};



//*******************************
//    Private Member Functions  *
//*******************************

/**
 * Initializes all the links on the lattice to the unit matrix
 */
__host__ void
LattiCuda::Initialize(){

        int half = h_size/4;
        dim3 in_Threads(4, 4, 4);
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

        int split = h_size/4;

        //Dimensions for the kernal
        dim3 Threads(4, 4, 4);
        dim3 Blocks(split, split, split);

        //All directions need to updated independently
        for(int d = 0; d < 4; d++) {

                //Checkerboard pattern for T dimension
                for(int offset = 0; offset <= 1; offset++) {
                        for(int tdim = 0; tdim < h_size/2; tdim++) {
                                GPU_Equilibrate<<<Blocks, Threads>>>(d_lattice, ((tdim)*2 + offset), d);
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

        int split = h_size/4;

        //Dimensions for the kernal
        dim3 Threads(4, 4, 4);
        dim3 Blocks(split, split, split);


        //Array to hold total avg plaquett per thread and total amount of iterations
        double *h_plaq;
        double *h_iter;

        h_plaq = new double[h_size*h_size*h_size*h_size];
        h_iter = new double[h_size*h_size*h_size*h_size];

        double *d_plaq;
        double *d_iter;

        cudaMalloc((void**)&d_plaq, sizeof(double)*h_size*h_size*h_size*h_size);
        cudaMalloc((void**)&d_iter, sizeof(double)*h_size*h_size*h_size*h_size);


        //Run on gpu for each time slice
        for(int tdim = 0; tdim < h_size; tdim++) {
                GPU_AvgPlaquette<<<Blocks, Threads>>>
                (d_lattice, tdim, d_plaq, d_iter);
        }
        cudaDeviceSynchronize();

        //Copy results from gpu
        cudaMemcpy(h_plaq, d_plaq, sizeof(double)*h_size*h_size*h_size*h_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_iter, d_iter, sizeof(double)*h_size*h_size*h_size*h_size, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        //Evaluate results
        double totplaq{0};
        double totiter{0};
        for(int i = 0; i < h_size*h_size*h_size*h_size; i++) {
                //cout << i << " "<< h_plaq[i] << "\n";
                totplaq += h_plaq[i];
                totiter += h_iter[i];
        }

        cudaFree(d_plaq);
        cudaFree(d_iter);
        delete[] h_plaq;
        delete[] h_iter;

        return (1 - totplaq/totiter);
};


/**
 * Calculates the expectation value of two polykov loops
 * @param  dist - Distance of loops
 * @return      - Average of all lattice locations
 */
__host__ double
LattiCuda::Polykov(int dist){

        int split = h_size/4;

        //Dimensions for the kernal
        dim3 Threads(4, 4, 4);
        dim3 Blocks(split, split, split);


        //Array to hold total avg plaquett per thread and total amount of iterations
        double *h_poly;
        double *h_iter;
        h_poly = new double[h_size*h_size*h_size];
        h_iter = new double[h_size*h_size*h_size];

        double *d_poly;
        double *d_iter;

        //Allocate GPU memory
        cudaMalloc((void**)&d_poly, sizeof(double)*h_size*h_size*h_size);
        cudaMalloc((void**)&d_iter, sizeof(double)*h_size*h_size*h_size);


        //Run on gpu for each time slice
        GPU_Polykov<<<Blocks, Threads>>>(d_lattice, d_poly, d_iter, dist);

        cudaDeviceSynchronize();

        //Copy results from gpu
        cudaMemcpy(h_poly, d_poly, sizeof(double)*h_size*h_size*h_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_iter, d_iter, sizeof(double)*h_size*h_size*h_size, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        //Evaluate results
        double totpoly{0};
        double totiter{0};
        for(int i = 0; i < h_size*h_size*h_size; i++) {
                //cout << i << " "<< h_plaq[i] << "\n";
                totpoly += h_poly[i];
                totiter += h_iter[i];
        }

        cudaFree(d_poly);
        cudaFree(d_iter);
        delete[] h_poly;
        delete[] h_iter;

        return totpoly/totiter;
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
        for( pos[0] = 0; pos[0] < h_size; pos[0]++) { // T dimension
                for( pos[1] = 0; pos[1] < h_size; pos[1]++) { // X dimension
                        for( pos[2] = 0; pos[2] < h_size; pos[2]++) { // Y dimension
                                for( pos[3] = 0; pos[3] < h_size; pos[3]++) { // Z dimension
                                        for(int d = 0; d < 4; d++) { // direction
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
LattiCuda::Load(){
        printf("Loading Lattice Configuration.......\n");

        fstream File;
        File.open("../Data/LatticeConfig.dat", ios::in );

        double real, imag;

        bool test = File.is_open();
        if(test) {
                int pos[4] = {0,0,0,0};
                for( pos[0] = 0; pos[0] < h_size; pos[0]++) { // T dimension
                        for( pos[1] = 0; pos[1] < h_size; pos[1]++) { // X dimension
                                for( pos[2] = 0; pos[2] < h_size; pos[2]++) { // Y dimension
                                        for( pos[3] = 0; pos[3] < h_size; pos[3]++) { // Z dimension
                                                for(int d = 0; d < 4; d++) { // direction
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

        }

        File.close();

        //Copy host lattice to device lattice
        cudaMemcpy(d_lattice, h_lattice, memsize*sizeof(bach::complex<double>), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        printf("Done Loading.\n");


};
