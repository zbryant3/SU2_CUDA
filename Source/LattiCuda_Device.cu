#include "./Headers/LattiCuda_Device.cuh"

#include <stdio.h>




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
LattiCuda_Device::MLoc(int *loc,int d, int m){

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

        coor = loc[1] + loc[2] * sharedcalc + loc[3] * sharedcalc * sharedcalc + loc[0] * sharedcalc * sharedcalc * sharedcalc
               + d * sharedcalc * sharedcalc * sharedcalc * 3 + m * sharedcalc * sharedcalc * sharedcalc * 3 * 4;

        return coor;

};



/**
 * Populates the sublattice based on the major lattice
 */
__device__ void
LattiCuda_Device::Populate(){


        //Fill the normal spots in all directions
        for(int d = 0; d < 4; d++) {
                //Matrix Fill
                for(int m = 0; m < 4; m++) {
                        SubLattice[SLoc(min, d, m)]
                                = Lattice[MLoc(maj, d, m)];
                }
        }
        __syncthreads();


        //Fill looking up in all dimensions
        for(int i = 0; i < 4; i++) {

                //Go up in i direction
                MU(min,i);
                MU(maj,i);

                //Look in all directions
                for(int d = 0; d < 4; d++) {
                        //Fill Matrix
                        for(int m = 0; m < 4; m++) {
                                SubLattice[SLoc(min, d, m)]
                                        = Lattice[MLoc(maj, d, m)];
                        }
                }

                //Go back down in original location
                MD(min,i);
                MD(maj,i);
                __syncthreads();
        }
        __syncthreads();


        //Fill looking down in all dimensions
        for(int i = 0; i < 4; i++) {

                //Go down in i direction
                MD(min,i);
                MD(maj,i);

                //Look in all directions
                for(int d = 0; d < 4; d++) {

                        //Fill Matrix
                        for(int m = 0; m < 4; m++) {
                                SubLattice[SLoc(min, d, m)]
                                        = Lattice[MLoc(maj, d, m)];
                        }
                }

                //Go back up in original location
                MU(min,i);
                MU(maj,i);
                __syncthreads();
        }
        __syncthreads();

};


/**
 * Equilibrates the lattice on thread based level
 */
__device__ void
LattiCuda_Device::ThreadEquilibrate(){


           thrust::complex<double> a[4], b[4], c[4];
           thrust::complex<double> w[4], w1[4], w2[4], w3[4];
           thrust::complex<double> v[4], v1[4], v2[4], v3[4];


           //Look in all directions
           for(int d = 0; d < 4; d++) {


                for(int j = 0; j < 4; j++) {
                        c[j] = thrust::complex<double>(0,0);
                }

                //For the directions other than that we are already looking in
                for(int i = 0; i < 4; i++) {
                        if(i != d) {

                                //Get link looking up in d
                                MU(min, d);
                                for(int k = 0; k < 4; k++)
                                        w1[k] = SubLattice[SLoc(min, i, k)];
                                MD(min, d);


                                //Get (hermitian conjugate) link looking up in i
                                MU(min, i);
                                for(int k = 0; k < 4; k++)
                                        w2[k] = thrust::conj(SubLattice[SLoc(min, d, (3 - k))]);
                                MD(min, i);

                                //Get (hermitian conjugate) link looking in i
                                for(int k = 0; k < 4; k++)
                                        w3[k] = thrust::conj(SubLattice[SLoc(min, i, (3 - k))]);
                                MU(min, d);

                                MD(min, i);
                                for(int k = 0; k < 4; k++)
                                        v1[k] = thrust::conj(SubLattice[SLoc(min, i, (3 - k))]);

                                MD(min, d);
                                for(int k = 0; k < 4; k++){

                                  v2[k] = thrust::conj(SubLattice[SLoc(min, d, (3 - k))]);
                                  v3[k] = SubLattice[SLoc(min, i, k)];

                                }


                        }
                }

           }


};


//*******************************
//    Public Member Functions   *
//*******************************

/**
 * Constructor for the Lattice QCD wrapper
 */
__device__
LattiCuda_Device::LattiCuda_Device(int* const_size, thrust::complex<double> *major_lattice,
                                   thrust::complex<double> *SubLatt, int t){

        size = const_size;

        Lattice = major_lattice;

        SubLattice = SubLatt;

        sharedcalc = (*size/ (*size/2) + 2);


        min[0] = 1;
        min[1] = threadIdx.x + 1;
        min[2] = threadIdx.y + 1;
        min[3] = threadIdx.z + 1;

        maj[0] = t;
        maj[1] = threadIdx.x + blockIdx.x * blockDim.x;
        maj[2] = threadIdx.y + blockIdx.y * blockDim.y;
        maj[3] = threadIdx.z + blockIdx.z * blockDim.z;

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

        //Major location of threads
        int major[4];
        major[1]  = threadIdx.x;
        major[2] = threadIdx.y + blockIdx.x * blockDim.y;
        major[3] = threadIdx.z + blockIdx.y * blockDim.z;
        major[0] = blockIdx.z;

        //Set links in all directions to the unit matrix
        for(int d = 0; d < 4; d++) {
                Lattice[MLoc(major,d,0)]
                        = thrust::complex<double>(1,0);
                Lattice[MLoc(major,d,1)]
                        = thrust::complex<double>(0,0);
                Lattice[MLoc(major,d,2)]
                        = thrust::complex<double>(0,0);
                Lattice[MLoc(major,d,3)]
                        = thrust::complex<double>(1,0);
        }

};


/**
 * Equilibrates the sublattices by populating the sublattices
 */
__device__ void
LattiCuda_Device::Equilibrate(){

        //Checkerboard pattern for blocks
        int Bremainder = (blockIdx.x + blockIdx.y + blockIdx.z)%2;

        //Checkerboard pattern for threads
        int Tremainder = (threadIdx.x + threadIdx.y + threadIdx.z)%2;

        if(Bremainder == 0) {
                Populate();

                if(Tremainder == 0) {
                        ThreadEquilibrate();
                }
                __syncthreads();

                if(Tremainder == 1) {
                        ThreadEquilibrate();
                }
                __syncthreads();
        }
        __syncthreads();

        if(Bremainder == 1) {
                Populate();

                if(Tremainder == 0) {
                        ThreadEquilibrate();
                }
                __syncthreads();

                if(Tremainder == 1) {
                        ThreadEquilibrate();
                }
                __syncthreads();
        }
        __syncthreads();

};



/**
 * Initiates a test for various reasons
 */
__device__ void
LattiCuda_Device::TestBack(){


};
