/**
 * Author: Zachariah Bryant
 * Description: Device based object to create in kernal
 *              to run operations on a SU(2) lattice.
 */


//  ********************
//  *      Headers     *
//  ********************

#ifndef LATTIDEVICE_H
#define LATTIDEVICE_H
#include "Complex.cuh"


//CUDA enabled random number generator
#include <cuda.h>
#include <curand_kernel.h>


//  ******************************
//  *      Class Declaration     *
//  ******************************

class LattiCuda_Device {
private:
int *size;
double *beta;
bach::complex<double> *Lattice;

// -1 since older version of bach cant handle multiplying by an integer
bach::complex<double> neg = bach::complex<double>(-1, 0);

// 0 - T  / 1 - X / 2 - Y / 3 - Z
int maj[4];

int tid;
curandStateMRG32k3a_t rng;

/**
 * Gets a 1D array location based on 4D SU2 parameters - FOR MAJOR LATTICE
 * @param  loc - Array for major lattice location
 * @param  d - direction of link
 * @param  m - matrix element
 * @return int  - array location for Lattice
 */
__device__ int
loc(int *loc,int d, int m);

/**
 * Moves down in a specific direction with periodic boundary
 * (CAUTION) Must move back UP in order to return to original dirction
 * @param  loc - Coordinate system to navigate
 * @param  d   - Direction to move in
 */
__device__ void
movedown(int *loc, int d);

/**
 * Moves up in a specific direction with periodic boundary
 * (CAUTION) Must move back DOWN in order to return to original dirction
 * @param  loc - Coordinate system to navigate
 * @param  d   - Direction to move in
 */
__device__ void
moveup(int *loc, int d);

/**
 * Gets the hermitian conjugate of a link
 * @param  pos - Array with lattice position
 * @param  d   - Direction to look in
 * @param  in  - Input matrix to save link to
 */
__device__ void
hermconj(int *pos, int d, bach::complex<double> *in);

/**
 * Gets a link from the major lattice
 * @param  pos - Major Lattice position to get link from
 * @param  d   - Direction to look in
 * @param  in  - Input matrix
 */
__device__ void
getlink(int *pos, int d, bach::complex<double> *in);

/**
 * Creates a random link based on the input matrix
 * @param  in  - Input Matrix
 * @param  out - Output Matrix
 */
__device__ void
randomlink(bach::complex<double> *in, bach::complex<double> *out);

/**
 * Generates a random integer from 0 to (t - 1)
 * @param  t - Bound for generation
 * @return int
 */
__device__ int
randomint(int t);

/**
 * Draws from reals from -1 to 1 or from 0 to 1
 * @param  t - t = 0 (-1 to 1) or t = anything else (0 to 1)
 * @return double
 */
__device__ double
randomdouble(int t);

/**
 * Equilibrates the lattice on a thread based level
 * @param  dir - Direction to equilibrate
 */
__device__ void
threadEquilibrate(int dir);


/**
 * Initializes the position for the sublattice and major lattice
 */
__device__ void
initialPos(int t);


/**
 * Multiplies two matrices together, saving the result to the third input
 * @param  m1 - Matrix 1
 * @param  m2 - Matrix 2
 * @param  r  - Result
 */
__device__ void
matrixMult(bach::complex<double> *m1, bach::complex<double> *m2, bach::complex<double> *r);

/**
 * Gets the Polykov Loop in a given position (matrix form)
 * @param  pos - Spatail position to look in
 * @param  in  - Input matrix to save product of matrices to
 */
__device__ void
polyloop(int *pos, bach::complex<double> *in);

public:

/**
 * Constructor for the Lattice QCD wrapper
 */
__device__
LattiCuda_Device(int *const_size, double *const_beta, bach::complex<double> *major_lattice, int t = 0);

/**
 * Destructor for the Lattice QCD wrapper
 */
__device__
~LattiCuda_Device();

/**
 * Equilibratess the lattice in a checkerboard pattern
 * @param   dir    - Direction to equilibrate
 */
__device__ void
equilibrate(int dir);

/**
 * Initializes all the links on the lattice to the unit matrix
 */
__device__ void
initialize();

/**
 * Generates the sum of plaquettes of the lattice configuration.
 * @param  plaq - Device memory for output of sum of plaquettes.
 * @param  iter - Number of plaquettes accounted for.
 */
__device__ void
avgPlaquette(double *plaq, double *iter);

/**
 * Sums the polykov loop at each lattice site multiplied by the
 * polykov loop in all spatial directions a set distance away
 * @param  poly - Array for each thread to save its sum to a unique location
 * @param  iter - Array for each thread to save its number of iterations
 * @param  dist - Distance to look in each spatial direction
 */
__device__ void
polykov(double *poly, double *iter, int dist);


};
#endif
