//***************************************************************************
//    Author: Zachariah Bryant
//    Function:
//
//***************************************************************************

#ifndef LATTIDEVICE_H
#define LATTIDEVICE_H

#include <thrust/complex.h>


//**************************
//    Class Declaration    *
//**************************
class LattiCuda_Device{
private:
  int *size;
  int sharedcalc;
  thrust::complex<double> *Lattice;
  thrust::complex<double> *SubLattice;

  // 0 - T  / 1 - X / 2 - Y / 3 - Z
  int min[4];
  int maj[4];
  int temp[4];

  int minX;
  int minY;
  int minZ;
  int minT;

  int majX;
  int majY;
  int majZ;
  int majT;


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

public:

  /**
   * Constructor for the Lattice QCD wrapper
   */
  __device__
  LattiCuda_Device(int *const_size, thrust::complex<double> *major_lattice,
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
   * Initiates a test for various reasons
   */
  __device__ void
  TestBack();


};
#endif
