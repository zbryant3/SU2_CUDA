/**
 * Author: Zachariah Bryant
 * Description: This is a class wrap for runnning SU(2) lattice qcd
 *              operations using CUDA.
 */


#ifndef LATTICUDA_H
#define LATTICUDA_H

//  ********************
//  *      Headers     *
//  ********************
#include "Complex.cuh"
#include <string>



//  *****************************
//  *      Kernal Functions     *
//  *****************************

/**
 * Initializes the lattice to unit matrices.
 * @param  d_lattice - Array to lattice in device memory.
 * @param  tdim      - Time dimension to initialize.
 */
__global__ void
gpu_Initialize(bach::complex<double> *d_lattice, int tdim);

/**
 * Equilibrates the lattice.
 * @param  d_lattice - Pointer to lattice in device memory.
 * @param  tdim      - Time dimension to equilibrate.
 * @param  dir       - Direction to equilibrate.
 */
__global__ void
gpu_Equilibrate(bach::complex<double> *d_lattice, int tdim, int dir);


/**
 * Gets the sum of plaquettes for the lattice.
 * @param  d_lattice - Pointer to lattice in device memory.
 * @param  tdim      - Time slice to look in.
 * @param  d_plaq    - Array to hold sum of plaquettes unique
 *                     for eaach lattice point.
 * @param  d_iter    - Amount of plaquettes counted unique for each lattice point.
 */
__global__ void
gpu_AvgPlaquette(bach::complex<double> *d_lattice, int tdim, double *d_plaq, double *d_iter);

/**
 * Generates the sum of traces of two polykov loops
 * @param  d_lattice - Pointer to lattice in device memory.
 * @param  d_poly    - Array holding the sum of the trace of two polykov loops.
 * @param  d_iter    - Amount of traces calculated.
 * @param  dist      - Distance of separation of the polykov loops.
 */
__global__ void
gpu_Polykov(bach::complex<double> *d_lattice, double *d_poly, double *d_iter, int dist);



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
  - Defined in .cu code since constant variables must be in a global scope
  __constant__ int d_size;
  __constant__ double d_beta;
  */


  /**
   * Initializes all lattice links to unit matrix by envoking kernal.
   */
  __host__
  void initialize();


  /**
   * Returns 1D array location for linearized 4D lattice
   * @param  dim - Array with lattice dimension location t,x,y,z
   * @param  d   - Direction to look in
   * @param  m   - Matrix element for link
   * @return     - Int for array location
   */
  __host__ int
  loc(int *dim, int d, int m);


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
   * Equilibrates the lattice by envoking the gpu kernal.
   */
  __host__ void
  equilibrate();


  /**
   * Generates the value of the average plaquette for the lattice.
   * @return double
   */
  __host__ double
  avgPlaquette();


  /**
  * Calculates the average of two polykov loops
  * @param  dist - Distance from
  * @return      [description]
  */
  __host__ double
  polykov(int dist);


  /**
   * Saves the lattice configuration to a file.
   */
  __host__ void
  save();


  /**
   * Loads a lattice configuration from a file
   */
  __host__ void
  load();



};

#endif
