//***************************************************************************
//    Author: Zachariah Bryant
//    Function: Perform lattice QCD operations utilizing the NVIDIA
//         CUDA. This is a wrapper for a GPU orientated class that
//         contains the main operations.
//
//***************************************************************************

#ifndef LATTICUDA_H
#define LATTICUDA_H

//*********************
//    Header Files    *
//*********************
#include "Complex.cuh"



//*********************************
//      GPU Kernal Functions      *
//*********************************

/**
 * Initializes all the links on the lattice to the unit matrix using GPU.
 * @param  d_lattice - Pointer to the lattice in device memory
 */
__global__ void
GPU_Initialize(bach::complex<double> *d_lattice, int tdim);

/**
 * Equilibrates the lattice using the GPU.
 * @param  d_lattice - Pointer to the lattice in device memory
 */
__global__ void
GPU_Equilibrate(bach::complex<double> *d_lattice, int tdim, int dir);


/**
 * Gets the average plaquette of the lattice
 */
__global__ void
GPU_AvgPlaquette(bach::complex<double> *d_lattice, int tdim, double *d_plaq, double *d_iter);



//**************************
//    Class Declaration    *
//**************************

class LattiCuda {
private:
  //Host Variables
  bach::complex<double> *h_lattice; //16 bytes per element
  int h_size;
  double h_beta;
  int memsize;


  //GPU Variables
  bach::complex<double> *d_lattice;

  /*  Constant GPU Variables
  - Defined in class code since constant variables must be in a global scope
  __constant__ int d_size;
  */


  /**
  * Initializes all the links on the lattice to the unit matrix
  */
  __host__
  void Initialize();


public:

  /**
   * Constructor for the Lattice QCD wrapper
   * @param   LattSize  - Size of desired lattice
   * @param   inBeta    - Beta value
   */
  __host__
  LattiCuda(int LattSize, double inBeta);


  /**
   * Destructor for the Lattice QCD wrapper
   */
  __host__
  ~LattiCuda();


  /**
   * Equilibrates the lattice
   */
  __host__ void
  Equilibrate();


  /**
   * Gets the value of the average plaquette of the lattice
   * @return double - Average Plaquette
   */
  __host__ double
  AvgPlaquette();



};

#endif
