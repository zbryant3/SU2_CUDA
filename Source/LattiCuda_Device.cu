/**
 * Author: Zachariah Bryant
 * Description: Device based object to create in kernal
 *              to run operations on a SU(2) lattice.
 */


//  ********************
//  *      Headers     *
//  ********************

#include "./Headers/LattiCuda_Device.cuh"
#include "./Headers/Complex.cuh"
#include <stdio.h>

//CUDA enabled random number generator
#include <cuda.h>
#include <curand_kernel.h>


//  *************************************
//  *      Private Member Functions     *
//  *************************************

/**
 * Moves down in a specific direction with periodic boundary
 * (CAUTION) Must move back UP in order to return to original dirction
 * @param  loc - Coordinate system to navigate
 * @param  d   - Direction to move in
 */
__device__ void
LattiCuda_Device::movedown(int *loc, int d){

        if((loc[d] - 1) < 0)
                loc[d] = (*size - 1);
        else
                loc[d] = (loc[d] - 1);
};


/**
 * Moves up in a specific direction with periodic boundary
 * (CAUTION) Must move back DOWN in order to return to original dirction
 * @param  loc - Coordinate system to navigate
 * @param  d   - Direction to move in
 */
__device__ void
LattiCuda_Device::moveup(int *loc, int d){

        if((loc[d] + 1) >= *size)
                loc[d] = 0;
        else
                loc[d] = (loc[d] + 1);
};


/**
 * Gets a 1D array location based on 4D SU2 parameters - FOR MAJOR LATTICE
 * @param  loc - Array for major lattice location
 * @param  d - direction of link
 * @param  m - matrix element
 * @return int  - array location for Lattice
 */
__device__ int
LattiCuda_Device::loc(int *loc, int d, int m){

        int coor{0};

        coor = loc[1] + loc[2]*(*size) + loc[3]*(*size)*(*size) + loc[0]*(*size)*(*size)*(*size)
               + d*(*size)*(*size)*(*size)*(*size) + m*(*size)*(*size)*(*size)*(*size)*(4);

        return coor;

};


/**
 * Initializes the position on the major lattice
 */
__device__ void
LattiCuda_Device::initialPos(int t){

        maj[0] = t;
        maj[1] = threadIdx.x + blockIdx.x * blockDim.x;
        maj[2] = threadIdx.y + blockIdx.y * blockDim.y;
        maj[3] = threadIdx.z + blockIdx.z * blockDim.z;

        tid = loc(maj, 0, 0);
};


/**
 * Multiplies two matrices together, saving the result to the third input
 * @param  m1 - Matrix 1
 * @param  m2 - Matrix 2
 * @param  r  - Result
 */
__device__ void
LattiCuda_Device::matrixMult(bach::complex<double> *m1, bach::complex<double> *m2, bach::complex<double> *r){

        r[0] = m1[0]*m2[0] + m1[1]*m2[2];
        r[1] = m1[0]*m2[1] + m1[1]*m2[3];
        r[2] = neg*bach::conj(r[1]);
        r[3] = bach::conj(r[0]);

};


/**
 * Generates a random integer from 0 to (t - 1)
 * @param  t - Bound for generation
 * @return int
 */
__device__ int
LattiCuda_Device::randomint(int t){

        return (curand(&rng) % t);

};


/**
 * Draws from reals from -1 to 1 or from 0 to 1
 * @param  t - t = 0 (-1 to 1) or t = anything else (0 to 1)
 * @return double
 */
__device__ double
LattiCuda_Device::randomdouble(int t){

        double z{0};
        if( t == 0 ) {
                z = 2*(curand_uniform_double(&rng)) - 1;
        }
        else{
                do
                        z = 1 - curand_uniform_double(&rng);
                while(z == 0);
        }

        return z;

};


/**
 * Gets the hermitian conjugate of a link
 * @param  pos - Array with lattice position
 * @param  d   - Direction to look in
 * @param  in  - Input matrix to save link to
 */
__device__ void
LattiCuda_Device::hermconj(int *pos, int d, bach::complex<double> *in){

        in[0] = bach::conj(Lattice[loc(pos, d, 0)]);
        in[1] = neg*Lattice[loc(pos, d, 1)];
        in[2] = bach::conj(Lattice[loc(pos, d, 1)]);
        in[3] = Lattice[loc(pos, d, 0)];
};


/**
 * Gets a link from the major lattice
 * @param  pos - Major Lattice position to get link from
 * @param  d   - Direction to look in
 * @param  in  - Input matrix
 */
__device__ void
LattiCuda_Device::getlink(int *pos, int d, bach::complex<double> *in){

        for(int m = 0; m < 4; m++) {
                in[m] = Lattice[loc(pos, d, m)];
        }

};


/**
 * Creates a random link based on the input matrix
 * @param  in  - Input Matrix
 * @param  out - Output Matrix
 */
__device__ void
LattiCuda_Device::randomlink(bach::complex<double> *in, bach::complex<double> *out){


        double sdet = sqrt((in[0]*in[3] - in[1]*in[2]).real());
        double y[4], r[4];

        //Normalize the input matrix
        sdet = sqrt(pow(bach::abs(in[0]),2) + pow(bach::abs(in[1]),2));
        in[0] = in[0]/sdet;
        in[1] = in[1]/sdet;
        in[2] = bach::conj(neg*in[1]);
        in[3] = bach::conj(in[0]);


        //Generate one acceptable number based on heatbath
        do {
                //Random number from (0,1)
                r[0] = randomdouble(1);

                // a0 = 1 + (1/B*k)ln(x)
                // exp[-2Bk] < x < 1
                y[0] = 1 + (1/( (*beta)*sdet ))*log( r[0]*(1-exp(-2*(*beta)*sdet)) + exp(-2*(*beta)*sdet));

                //Random number from (0,1)
                r[0] = randomdouble(1);

        } while(pow(y[0], 2) > 1 - pow(r[0], 2)); // a0^2 > 1 - r^2



        //Generate 3 random numbers to be used to generate final matrix
        do {

                for(int i = 1; i < 4; i++) {
                        r[i] = randomdouble(0);
                }

        } while( (pow(r[1], 2) + pow(r[2], 2) + pow(r[3], 2)) > 1 );


        //Use the three randomly generated numbers to generate the matrix elements
        for(int i = 1; i < 4; i++)
                y[i] = sqrt(1 - pow(y[0],2)) * (r[i]/sqrt(pow(r[1],2) + pow(r[2],2) + pow(r[3],2)));

        bach::complex<double> m[4], w[4];


        m[0] = bach::complex<double>(y[0], y[3]);
        m[1] = bach::complex<double>(y[2], y[1]);
        m[2] = bach::complex<double>((-1)*y[2], y[1]);
        m[3] = bach::complex<double>(y[0], (-1)*y[3]);


        //Get the hermition conjugate of the input matrix
        w[0] = bach::conj(in[0]);
        w[1] = neg*in[1];
        w[2] = bach::conj(in[1]);
        w[3] = in[0];


        //Multiply the generated matrix and the hermition conjugate
        //And save to the output matrix
        matrixMult(m, w, out);

        /*
           //Normalize the new link - doesn't change anything up to .15 precision
           sdet = sqrt(pow(bach::abs(out[0]),2) + pow(bach::abs(out[1]),2));
           out[0] = out[0]/sdet;
           out[1] = out[1]/sdet;
           out[2] = bach::conj(neg*out[1]);
           out[3] = bach::conj(out[0]);
         */

};


/**
 * Equilibrates the lattice on a thread based level
 * @param  d - Direction to equilibrate
 */
__device__ void
LattiCuda_Device::threadEquilibrate(int d) {


        bach::complex<double> a[4], b[4], c[4];
        bach::complex<double> w[4], w1[4], w2[4], w3[4];
        bach::complex<double> v[4], v1[4], v2[4], v3[4];

        int pos[4];

        //Sets random seed
        curand_init(clock64(), loc(maj, d, 0), 0, &rng);


        //Go to initial position
        for(int j = 0; j < 4; j++)
                pos[j] = maj[j];

        //Initialize c
        for(int j = 0; j < 4; j++) {
                c[j] = bach::complex<double>(0,0);
        }


        //Get staple
        for(int i = 0; i < 4; i++) {
                if(i != d) {

                        //Move up in d and get w1 in i
                        moveup(pos, d);
                        for(int m = 0; m < 4; m++) {
                                w1[m] = Lattice[loc(pos, i, m)];
                        }



                        //Move down in i and get v1 in i (hermitian conj)
                        movedown(pos, i);
                        hermconj(pos, i, v1);

                        //Move down in d and get v2 in d (hermitian conj)
                        // and get v3 in i
                        movedown(pos, d);
                        hermconj(pos, d, v2);
                        for(int m = 0; m < 4; m++) {
                                v3[m] = Lattice[loc(pos, i, m)];
                        }
                        //Move up in i (original location now)
                        // and get w3 in i (hermitian conj)
                        moveup(pos, i);
                        hermconj(pos, i, w3);

                        //Move up in i and get w2 in d (hermitian conj)
                        // then move back down to original location
                        moveup(pos, i);
                        hermconj(pos, d, w2);
                        movedown(pos, i);



                        //Multiply matrices w2xw3 and v2xv3
                        matrixMult(w2, w3, w);
                        matrixMult(v2, v3, v);




                        //Multiply w1xw and v1, v and add the results
                        matrixMult(w1, w, a);
                        matrixMult(v1, v, b);

                        for(int k = 0; k < 4; k++) {
                                c[k] = c[k] + (a[k] + b[k]);
                        }

                        /*
                           if(maj[0] == 0 && maj[1] == 0 && maj[2] == 0 && maj[3] == 0){
                           for(int m = 0; m < 4; m++){
                            bach::print(c[m]);
                           }
                           double dete = (c[0]*c[3] - c[1]*c[2]).real();
                           printf("DET: %.15g\n", dete);
                           printf("\n");
                           }
                         */

                }

        }

        //Get a randomly generated link matrix based on C and save to A
        randomlink(c, a);

        //Save Randomized Link
        for(int k = 0; k < 4; k++)
                Lattice[loc(maj, d, k)] = a[k];

};


/**
 * Gets the Polykov Loop in a given position (matrix form)
 * @param  pos - Spatail position to look in
 * @param  in  - Input matrix to save product of matrices to
 */
__device__ void
LattiCuda_Device::polyloop(int *pos, bach::complex<double> *in){


        bach::complex<double> w1[4], w2[4], w3[4];

        //Make sure we are at 0 time location
        pos[0] = 0;

        //Get first link
        getlink(pos, 0, w1);

        //Move up in time, getting the links and then multiplying them
        for(int k = 1; k < *size; k++) {

                pos[0] = k;

                getlink(pos, 0, w2);

                matrixMult(w1, w2, w3);

                for(int m = 0; m < 4; m++) {
                        w1[m] = w3[m];
                }

        }

        for(int i = 0; i < 4; i++) {
                in[i] = w1[i];
        }
};


//  ************************************
//  *      Public Member Functions     *
//  ************************************

/**
 * Constructor for the Lattice QCD wrapper
 */
__device__
LattiCuda_Device::LattiCuda_Device(int* const_size, double *const_beta, bach::complex<double> *major_lattice, int t){

        size = const_size;

        beta = const_beta;

        Lattice = major_lattice;

        initialPos(t);

        tid = loc(maj, 0, 0);

};


/**
 * Destructor for the Lattice QCD wrapper
 */
__device__
LattiCuda_Device::~LattiCuda_Device(){


};


/**
 * Initializes all the links on the lattice to the unit matrix
 */
__device__ void
LattiCuda_Device::initialize(){

        //Set links in all directions to the unit matrix
        for(int d = 0; d < 4; d++) {
                Lattice[loc(maj,d,0)]
                        = bach::complex<double>(1,0);
                Lattice[loc(maj,d,1)]
                        = bach::complex<double>(0,0);
                Lattice[loc(maj,d,2)]
                        = bach::complex<double>(0,0);
                Lattice[loc(maj,d,3)]
                        = bach::complex<double>(1,0);

        }
        __syncthreads();

};


/**
 * Equilibratess the lattice in a checkerboard pattern
 * @param   dir    - Direction to equilibrate
 */
__device__ void
LattiCuda_Device::equilibrate(int dir){


        //Checkerboard pattern for blocks
        int Bremainder = (blockIdx.x + blockIdx.y + blockIdx.z)%2;

        //Checkerboard pattern for threads
        int Tremainder = (threadIdx.x + threadIdx.y + threadIdx.z)%2;

        if(Bremainder == 0 && Tremainder == 0) {
                threadEquilibrate(dir);
        }
        __syncthreads();


        if(Bremainder == 0 && Tremainder == 1) {
                threadEquilibrate(dir);
        }
        __syncthreads();


        if(Bremainder == 1 && Tremainder == 0) {
                threadEquilibrate(dir);
        }
        __syncthreads();


        if(Bremainder == 1 && Tremainder == 1) {
                threadEquilibrate(dir);
        }
        __syncthreads();
};


/**
 * Generates the sum of plaquettes of the lattice configuration.
 * @param  plaq - Device memory for output of sum of plaquettes.
 * @param  iter - Number of plaquettes accounted for.
 */
__device__ void
LattiCuda_Device::avgPlaquette(double *plaq, double *iter){

        bach::complex<double> w[4], w1[4], w2[4], w3[4], w4[4];
        bach::complex<double> v[4];
        int pos[4];

        plaq[tid] = 0;
        iter[tid] = 0;


        for(int d = 0; d < 4; d++)
        {
                for(int i = 0; i < d; i++)
                {
                        if(i != d) {

                                //Get current position
                                for(int k = 0; k < 4; k++)
                                        pos[k] = maj[k];

                                //Get link in direction of d
                                for(int m = 0; m < 4; m++)
                                        w1[m] = Lattice[loc(pos, d, m)];


                                //Look up in direction of d and get link of i
                                moveup(pos, d);
                                for(int m = 0; m < 4; m++)
                                        w2[m] = Lattice[loc(pos, i, m)];
                                movedown(pos, d);

                                //Look up in direction of i and get link of d (conjugated)
                                moveup(pos, i);
                                hermconj(pos, d, w3);
                                movedown(pos, i);

                                //Get link in direction of dir2 (conjugated)
                                hermconj(pos, i, w4);

                                matrixMult( w1, w2, w);
                                matrixMult( w3, w4, v);

                                matrixMult( w, v, w1);


                                bach::complex<double> temp = (w1[0] + w1[3]);

                                plaq[tid] += 0.5*temp.real();
                                iter[tid] += 1;

                        }

                }
        }

};


/**
 * Sums the polykov loop at each lattice site multiplied by the
 * polykov loop in all spatial directions a set distance away
 * @param  poly - Array for each thread to save its sum to a unique location
 * @param  iter - Array for each thread to save its number of iterations
 * @param  dist - Distance to look in each spatial direction
 */
__device__ void
LattiCuda_Device::polykov(double *poly, double *iter, int dist){

        bach::complex<double> p1[4], p2[4];

        //Initialize summation for thread location
        poly[tid] = 0;
        iter[tid] = 0;

        //Get the position of the thread
        int pos[4];
        for(int i = 0; i < 4; i++) {
                pos[i] = maj[i];
        }

        //Looking in all spatial dimensions
        for(int dir = 1; dir < 4; dir++) {

                //   ***First Temporal Transporter***
                polyloop(pos, p1);


                //   ***Second Temporal Transporter***

                //Move up to position by a set distance
                for(int i = 1; i <= dist; i++) {
                        moveup(pos, dir);
                }

                polyloop(pos, p2);

                //Move back down to original position
                for(int i = 0; i < 4; i++) {
                        pos[i] = maj[i];
                }

                //Get the hermitian conjugate of the second temporal transporter
                p2[3] = p2[0];
                p2[2] = bach::conj(p2[1]);
                p2[1] = neg*p2[1];
                p2[0] = bach::conj(p2[0]);

                poly[tid] += (p1[0] + p1[3]).real() * (p2[0] + p2[3]).real();
                iter[tid] += 1;
        }


};
