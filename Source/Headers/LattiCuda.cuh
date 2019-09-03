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
#include <string>



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

/**
 * Gets the trace of two polykov loops on all lattice sites of a set distance
 * @param d_lattice [description]
 * @param d_plaq    [description]
 * @param d_iter    [description]
 * @param dist      [description]
 */
__global__ void
GPU_Polykov(bach::complex<double> *d_lattice, double *d_poly, double *d_iter, int dist);



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


  /**
   * Returns 1D array location for linearized 4D lattice
   * @param  dim - Array with lattice dimension location t,x,y,z
   * @param  d   - Direction to look in
   * @param  m   - Matrix element for link
   * @return     - Int for array location
   */
  __host__ int
  Loc(int *dim, int d, int m);


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
   * Calculates the average of two polykov loops
   * @param  dist - Distance from
   * @return      [description]
   */
  __host__ double
  Polykov(int dist);


  /**
   * Gets the value of the average plaquette of the lattice
   * @return double - Average Plaquette
   */
  __host__ double
  AvgPlaquette();

  /**
   * Saves the lattice configuration to a file.
   */
  __host__ void
  Save();


  /**
   * Loads a lattice configuration from a file
   * @param  file - File to load from
   */
  __host__ void
  Load();



};

#endif
