#include "./Headers/LattiCuda_Device.cuh"

#include <stdio.h>

//CUDA enabled random number generator
#include <cuda.h>
#include <curand_kernel.h>




//*******************************
//    Private Member Functions  *
//*******************************



/**
 * Moves down in a specific direction with periodic boundary
 * (CAUTION) Must move back UP in order to return to original dirction
 * @param  loc - Coordinate system to navigate
 * @param  d   - Direction to move in
 */
__device__ void
LattiCuda_Device::MD(int *loc, int d){

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
LattiCuda_Device::MU(int *loc, int d){

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
LattiCuda_Device::MLoc(int *loc, int d, int m){

        int coor{0};

        coor = loc[1] + loc[2]*(*size) + loc[3]*(*size)*(*size) + loc[0]*(*size)*(*size)*(*size)
               + d*(*size)*(*size)*(*size)*(*size) + m*(*size)*(*size)*(*size)*(*size)*(4);

        return coor;

};


/**
 * Gets a 1D array location based on 4D SU2 parameters - FOR SUBLATTICE
 * @param  loc - Array for minor lattice location
 * @param  d - direction of link
 * @param  m - matrix element
 * @return int  - array location for Lattice
 */
__device__ int
LattiCuda_Device::SLoc(int *loc, int d, int m){

        //sharedcalc = (*size/ (*size/2) + 2); done on creation of model

        int coor{0};

        coor = loc[1] + loc[2]*sharedcalc + loc[3]*sharedcalc*sharedcalc + loc[0]*sharedcalc*sharedcalc*sharedcalc
               + d*sharedcalc*sharedcalc*sharedcalc*3 + m*sharedcalc*sharedcalc*sharedcalc*3*4;

        return coor;

};



/**
 * Populates the sublattice based on the major lattice
 */
__device__ void
LattiCuda_Device::Populate(){

        //Traverse time directions
        for(int t = 0; t < 3; t++) {
                min[0] = t;

                for(int d = 0; d < 4; d++) {
                        //Fill normal positions
                        for(int m = 0; m < 4; m++) {
                                SubLattice[SLoc(min, d, m)]
                                        = Lattice[MLoc(maj, d, m)];
                        }
                }

                //Fill looking up in the X direction
                if(min[1] == blockDim.x) {

                        //Move up in X
                        MU(min, 1);
                        MU(maj, 1);

                        //Matrix Fill
                        for(int d = 0; d < 4; d++) {
                                for(int m = 0; m < 4; m++) {
                                        SubLattice[SLoc(min, d, m)]
                                                = Lattice[MLoc(maj, d, m)];
                                }
                        }

                        //Move down in X
                        MD(min, 1);
                        MD(maj, 1);
                }
                __syncthreads();

                //Fill looking Down in the X direction
                if(min[1] == 1) {

                        //Move down in X
                        MD(min, 1);
                        MD(maj, 1);

                        //Matrix Fill
                        for(int d = 0; d < 4; d++) {
                                for(int m = 0; m < 4; m++) {
                                        SubLattice[SLoc(min, d, m)]
                                                = Lattice[MLoc(maj, d, m)];
                                }
                        }
                        //Move up in X
                        MU(min, 1);
                        MU(maj, 1);
                }
                __syncthreads();

                //Fill looking up in the Y direction
                if(min[2] == blockDim.y) {

                        //Move up in Y
                        MU(min, 2);
                        MU(maj, 2);

                        //Fill normall positions
                        for(int d = 0; d < 4; d++) {
                                for(int m = 0; m < 4; m++) {
                                        SubLattice[SLoc(min, d, m)]
                                                = Lattice[MLoc(maj, d, m)];
                                }
                        }

                        //Move down in Y
                        MD(min, 2);
                        MD(maj, 2);
                }
                __syncthreads();

                //Fill looking Down in the Y direction
                if(min[2] == 1) {

                        //Move down in Y
                        MD(min, 2);
                        MD(maj, 2);

                        //Fill normal positions
                        for(int d = 0; d < 4; d++) {
                                for(int m = 0; m < 4; m++) {
                                        SubLattice[SLoc(min, d, m)]
                                                = Lattice[MLoc(maj, d, m)];
                                }
                        }

                        //Move up in Y
                        MU(min, 2);
                        MU(maj, 2);
                }
                __syncthreads();

                //Fill looking up in the Z direction
                if(min[3] == blockDim.z) {

                        //Move up in Z
                        MU(min, 3);
                        MU(maj, 3);



                        //Fill normal positions
                        for(int d = 0; d < 4; d++) {
                                for(int m = 0; m < 4; m++) {
                                        SubLattice[SLoc(min, d, m)]
                                                = Lattice[MLoc(maj, d, m)];
                                }
                        }

                        //Move down in Z
                        MD(min, 3);
                        MD(maj, 3);
                }
                __syncthreads();

                //Fill looking Down in the Z direction
                if(min[3] == 1) {

                        //Move down in Z
                        MD(min, 3);
                        MD(maj, 3);

                        //Fill normal positions
                        for(int d = 0; d < 4; d++) {
                                for(int m = 0; m < 4; m++) {
                                        SubLattice[SLoc(min, d, m)]
                                                = Lattice[MLoc(maj, d, m)];
                                }
                        }

                        //Move up in Z
                        MU(min, 3);
                        MU(maj, 3);
                }
                __syncthreads();


        }




};


/**
 * Initializes the position for the sublattice and major lattice
 */
__device__ void
LattiCuda_Device::IniPos(int t){

        min[0] = 1;
        min[1] = threadIdx.x + 1;
        min[2] = threadIdx.y + 1;
        min[3] = threadIdx.z + 1;

        maj[0] = t;
        maj[1] = threadIdx.x + blockIdx.x * blockDim.x;
        maj[2] = threadIdx.y + blockIdx.y * blockDim.y;
        maj[3] = threadIdx.z + blockIdx.z * blockDim.z;

        tid = MLoc(maj, 0, 0);
};



/**
 * Multiplies two matrices together, saving the result to the third input
 * @param  m1 - Matrix 1
 * @param  m2 - Matrix 2
 * @param  r  - Result
 */
__device__ void
LattiCuda_Device::MaMult(thrust::complex<double> *m1, thrust::complex<double> *m2, thrust::complex<double> *r){

        r[0] = m1[0] * m2[0] + m1[1] * m2[2];
        r[1] = m1[0] * m2[1] + m1[1] * m2[3];
        r[2] = (-1)*thrust::conj(r[1]);
        r[3] = thrust::conj(r[0]);

};



/**
 * Generates a random integer from 0 to (t - 1)
 * @param  t - Bound for generation
 * @return int
 */
__device__ int
LattiCuda_Device::RandInt(int t){

        return (curand(&rng) % t);

};

/**
 * Draws from reals from -1 to 1 or from 0 to 1
 * @param  t - t = 0 (-1 to 1) or t = anything else (0 to 1)
 * @return double
 */
__device__ double
LattiCuda_Device::RandDouble(int t){

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



__device__ void
LattiCuda_Device::HermConj(int *pos, int d, thrust::complex<double> *in){

        in[0] = thrust::conj(Lattice[MLoc(pos, d, 0)]);
        in[1] = (-1)*Lattice[MLoc(pos, d, 1)];
        in[2] = thrust::conj(Lattice[MLoc(pos, d, 1)]);
        in[3] = Lattice[MLoc(pos, d, 0)];
};


/**
 * Creates a random link based on the input matrix
 * @param  in  - Input Matrix
 * @param  out - Output Matrix
 */
__device__ void
LattiCuda_Device::RandLink(thrust::complex<double> *in, thrust::complex<double> *out){


        thrust::complex<double> temp = (in[0]*in[3] - in[1]*in[2]);

        double sdet = sqrt(temp.real());
        double y[4], r[4];


        /*
           //Normalize the input matrix
           for(int i = 0; i < 4; i++)
               in[i] = in[i] / sdet;
         */

        sdet = sqrt(pow(thrust::abs(in[0]),2) + pow(thrust::abs(in[1]),2));
        in[0] = in[0]/sdet;
        in[1] = in[1]/sdet;
        in[2] = thrust::conj((-1)*in[1]);
        in[3] = thrust::conj(in[0]);






        //Generate one acceptable number based on heatbath
        do {
                //Random number from 0-1
                r[0] = RandDouble(1);


                // a0 = 1 + (1/B*k)ln(x)
                // exp[-2Bk] < x < 1
                y[0] = 1 + (1/( (*beta)*sdet ))*log( r[0]*(1-exp(-2*(*beta)*sdet)) + exp(-2*(*beta)*sdet));

                //Random number from (0,1)
                r[0] = RandDouble(1);

        } while(pow(y[0], 2) > 1 - pow(r[0], 2)); // a0^2 > 1 - r^2



        //Generate 3 random numbers to be used to generate final matrix
        do {

                for(int i = 1; i < 4; i++) {
                        r[i] = RandDouble(0);
                }

        } while( (pow(r[1], 2) + pow(r[2], 2) + pow(r[3], 2)) > 1 );


        //Use the three randomly generated numbers to generate the matrix elements
        for(int i = 1; i < 4; i++)
                y[i] = sqrt(1 - pow(y[0],2)) * (r[i]/sqrt(pow(r[1],2) + pow(r[2],2) + pow(r[3],2)));

        thrust::complex<double> m[4], w[4];

        m[0] = thrust::complex<double>(y[0], y[3]);
        m[1] = thrust::complex<double>(y[2], y[1]);
        m[2] = thrust::complex<double>((-1)*y[2], y[1]);
        m[3] = thrust::complex<double>(y[0], (-1)*y[3]);

        //Get the hermition conjugate of the input matrix
        w[0] = thrust::conj(in[0]);
        w[1] = (-1)*in[1];
        w[2] = thrust::conj(in[1]);
        w[3] = in[0];

        //Multiply the generated matrix and the hermition conjugate
        //And save to the output matrix
        MaMult(m, w, out);

        sdet = sqrt(pow(thrust::abs(out[0]),2) + pow(thrust::abs(out[1]),2));
        out[0] = out[0]/sdet;
        out[1] = out[1]/sdet;
        out[2] = thrust::conj((-1)*out[1]);
        out[3] = thrust::conj(out[0]);

};


/**
 * Equilibrates the lattice on thread based level
 */
__device__ void
LattiCuda_Device::ThreadEquilibrate(int d) {

        thrust::complex<double> a[4], b[4], c[4];
        thrust::complex<double> w[4], w1[4], w2[4], w3[4];
        thrust::complex<double> v[4], v1[4], v2[4], v3[4];

        int pos[4];

        //Sets random seed
        curand_init(clock64(), MLoc(maj, d, 0), 0, &rng);


        //Go to initial position
        for(int j = 0; j < 4; j++)
                pos[j] = maj[j];

        //Initialize C
        for(int j = 0; j < 4; j++) {
                c[j] = thrust::complex<double>(0,0);
        }


        //For the directions other than that we are already looking in
        for(int i = 0; i < 4; i++) {
                if(i != d) {

                        //Move up in d and get w1 in i
                        MU(pos, d);
                        for(int m = 0; m < 4; m++) {
                                w1[m] = Lattice[MLoc(pos, i, m)];
                        }

                        //Move down in i and get v1 in i (hermitian conj)
                        MD(pos, i);
                        HermConj(pos, i, v1);

                        //Move down in d and get v2 in d (hermitian conj)
                        // and get v3 in i
                        MD(pos, d);
                        HermConj(pos, d, v2);
                        for(int m = 0; m < 4; m++) {
                                v3[m] = Lattice[MLoc(pos, i, m)];
                        }
                        //Move up in i (original location now)
                        // and get w3 in i (hermitian conj)
                        MU(pos, i);
                        HermConj(pos, i, w3);



                        //Move up in i and get w2 in d (hermitian conj)
                        // then move back down to original location
                        MU(pos, i);
                        HermConj(pos, d, w2);
                        MD(pos, i);


                        //Multiply matrices w2xw3 and v2xv3
                        MaMult(w2, w3, w);
                        MaMult(v2, v3, v);


                        //Multiply w1xw and v1, v and add the results
                        MaMult(w1, w, a);
                        MaMult(v1, v, b);


                        for(int k = 0; k < 4; k++) {
                                c[k] += (a[k] + b[k]);
                        }

                }

        }

        //Get a randomly generated link matrix based on C and save to A
        RandLink(c, a);

        //Save Randomized Link
        for(int k = 0; k < 4; k++)
                Lattice[MLoc(maj, d, k)] = a[k];

};


//*******************************
//    Public Member Functions   *
//*******************************

/**
 * Constructor for the Lattice QCD wrapper
 */
__device__
LattiCuda_Device::LattiCuda_Device(int* const_size, double *const_beta, thrust::complex<double> *major_lattice,
                                   thrust::complex<double> *SubLatt, int t){

        size = const_size;

        beta = const_beta;

        Lattice = major_lattice;

        SubLattice = SubLatt;

        sharedcalc = 4;

        IniPos(t);

        tid = MLoc(maj, 0, 0);




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
LattiCuda_Device::Initialize(){

        //Set links in all directions to the unit matrix
        for(int d = 0; d < 4; d++) {
                Lattice[MLoc(maj,d,0)]
                        = thrust::complex<double>(1,0);
                Lattice[MLoc(maj,d,1)]
                        = thrust::complex<double>(0,0);
                Lattice[MLoc(maj,d,2)]
                        = thrust::complex<double>(0,0);
                Lattice[MLoc(maj,d,3)]
                        = thrust::complex<double>(1,0);

        }
        __syncthreads();

};


/**
 * Equilibrates the sublattices by populating the sublattices
 */
__device__ void
LattiCuda_Device::Equilibrate(int dir){


        //Checkerboard pattern for blocks
        int Bremainder = (blockIdx.x + blockIdx.y + blockIdx.z)%2;

        //Checkerboard pattern for threads
        int Tremainder = (min[1] + min[2] + min[3])%2;

        if(Bremainder == 0 && Tremainder == 0) {
                ThreadEquilibrate(dir);
        }
        __syncthreads();


        if(Bremainder == 0 && Tremainder == 1) {
                ThreadEquilibrate(dir);
        }
        __syncthreads();


        if(Bremainder == 1 && Tremainder == 0) {
                ThreadEquilibrate(dir);
        }
        __syncthreads();


        if(Bremainder == 1 && Tremainder == 1) {
                ThreadEquilibrate(dir);
        }
        __syncthreads();
};


/**
 * Generates the average plaquette for each block
 */
__device__ void
LattiCuda_Device::AvgPlaquette(double *plaq, double *iter){

        //Populate();

        thrust::complex<double> w[4], w1[4], w2[4], w3[4], w4[4];
        thrust::complex<double> v[4];
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
                                        w1[m] = Lattice[MLoc(pos, d, m)];

                                //Look up in direction of d and get link of i
                                MU(pos, d);
                                for(int m = 0; m < 4; m++)
                                        w2[m] = Lattice[MLoc(pos, i, m)];
                                MD(pos, d);

                                //Look up in direction of i and get link of d (conjugated)
                                MU(pos, i);
                                HermConj(pos, d, w3);
                                MD(pos, i);

                                //Get link in direction of dir2 (conjugated)
                                HermConj(pos, i, w4);

                                MaMult( w1, w2, w);
                                MaMult( w3, w4, v);

                                MaMult( w, v, w1);

                                thrust::complex<double> temp = (w1[0] + w1[3]);

                                plaq[tid] += 0.5*temp.real();
                                iter[tid] += 1;
                        }

                }
        }

};
