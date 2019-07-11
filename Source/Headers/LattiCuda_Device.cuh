//***************************************************************************
//    Author: Zachariah Bryant
//    Function:
//
//***************************************************************************

#ifndef LATTIDEVICE_H
#define LATTIDEVICE_H

#include <thrust/complex.h>


//CUDA enabled random number generator
#include <cuda.h>
#include <curand_kernel.h>


//**************************
//    Class Declaration    *
//**************************
class LattiCuda_Device{
private:
  int *size;
  double *beta;
  int sharedcalc;
  thrust::complex<double> *Lattice;
  thrust::complex<double> *SubLattice;

  // 0 - T  / 1 - X / 2 - Y / 3 - Z
  int min[4];
  int maj[4];

  int tid;
  curandStatePhilox4_32_10_t rng;


  /**
   * Gets a 1D array location based on 4D SU2 parameters - FOR MAJOR LATTICE
   * @param  loc - Array for major lattice location
   * @param  d - direction of link
   * @param  m - matrix element
   * @return int  - array location for Lattice
   */
  __device__ int
  MLoc(int *loc,int d, int m);


  /**
   * Gets a 1D array location based on 4D SU2 parameters - FOR SUBLATTICE
   * @param  loc - Array for minor lattice location
   * @param  d - direction of link
   * @param  m - matrix element
   * @return int  - array location for Lattice
   */
  __device__ int
  SLoc(int *loc, int d, int m);


  /**
   * Moves down in a specific direction with periodic boundary
   * (CAUTION) Must move back UP in order to return to original dirction
   * @param  loc - Coordinate system to navigate
   * @param  d   - Direction to move in
   */
  __device__ void
  MD(int *loc, int d);



  /**
   * Moves up in a specific direction with periodic boundary
   * (CAUTION) Must move back DOWN in order to return to original dirction
   * @param  loc - Coordinate system to navigate
   * @param  d   - Direction to move in
   */
  __device__ void
  MU(int *loc, int d);


  __device__ void
  HermConj(int *pos, int d, thrust::complex<double> *in);

  /**
   * Creates a random link based on the input matrix
   * @param  in  - Input Matrix
   * @param  out - Output Matrix
   */
  __device__ void
  RandLink(thrust::complex<double> *in, thrust::complex<double> *out);

  /**
   * Generates a random integer from 0 to (t - 1)
   * @param  t - Bound for generation
   * @return int
   */
  __device__ int
  RandInt(int t);


  /**
   * Draws from reals from -1 to 1 or from 0 to 1
   * @param  t - t = 0 (-1 to 1) or t = anything else (0 to 1)
   * @return double
   */
  __device__ double
  RandDouble(int t);



  /**
   * Populates the sublattice based on the major lattice
   */
  __device__ void
  Populate();


  /**
   * Equilibrates the lattice on thread based level
   */
  __device__ void
  ThreadEquilibrate();


  /**
   * Initializes the position for the sublattice and major lattice
   */
  __device__ void
  IniPos(int t);


  /**
   * Multiplies two matrices together, saving the result to the third input
   * @param  m1 - Matrix 1
   * @param  m2 - Matrix 2
   * @param  r  - Result
   */
  __device__ void
  MaMult(thrust::complex<double> *m1, thrust::complex<double> *m2, thrust::complex<double> *r);

public:

  /**
   * Constructor for the Lattice QCD wrapper
   */
  __device__
  LattiCuda_Device(int *const_size, double *const_beta, thrust::complex<double> *major_lattice,
     thrust::complex<double> *SubLatt = NULL, int t = 0);

  /**
   * Destructor for the Lattice QCD wrapper
   */
  __device__
  ~LattiCuda_Device();

  /**
   * Equilibrates the sublattices by populating the sublattices
   */
  __device__ void
  Equilibrate();


  /**
  * Initializes all the links on the lattice to the unit matrix
  */
  __device__ void
  Initialize();


  /**
   * Generates the average plaquette for each block
   */
  __device__ void
  AvgPlaquette(double *plaq, double *iter);


};
#endif
