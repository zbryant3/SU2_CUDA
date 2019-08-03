//******************************************************************
//    Author: Zachariah Bryant
//    Function: Object that creates a complex number.
//
//******************************************************************

#ifndef COMPLEX_H
#define COMPLEX_H


#include <stdio.h>
#include <stdlib.h>

namespace bach {

template <class T>
class complex {
private:
T x, y;


public:

/**
 * Constructor for the complex number
 * @param  inreal - Real part
 * @param  inimag - Imaginary part
 */
__host__ __device__
complex(T inreal = 0, T inimag = 0) : x(inreal), y(inimag) {
};


/**
 * Destructor for the complex number
 */
__host__ __device__
~complex() {
};

/**
 * Returns the real part of the complex number
 */
__host__ __device__
T real(){
        return x;
};

/**
 * Returns the imaginary part of the complex number
 */
__host__ __device__
T imag(){
        return y;
};



/**
 * Operator for adding
 */
__host__ __device__
friend complex<T> operator+(const complex<T>& lhs, const complex<T>& rhs){
        return complex<T> (lhs.x + rhs.x, lhs.y + rhs.y);
};


/**
 * Operator for subtracting
 */
__host__ __device__
friend complex<T> operator-(const complex<T>& lhs,const complex<T>& rhs){
        return complex<T> (lhs.x - rhs.x, lhs.y - rhs.y);
};


/**
 * Operator for multiplying
 */
__host__ __device__
friend complex<T> operator*(const complex<T>& lhs, const complex<T>& rhs){
        double r,i;
        r = lhs.x*rhs.x - lhs.y*rhs.y;
        i = lhs.x*rhs.y + lhs.y*rhs.x;
        return complex<T>(r,i);
};


/**
 * Dividing by a complex number
 */
__host__ __device__
friend complex<T> operator/(const complex<T>& lhs, const complex<T>& rhs){
        double r,i,d;
        d = rhs.x*rhs.x + rhs.y*rhs.y;
        r = lhs.x*rhs.x + lhs.y*rhs.y;
        i = lhs.y*rhs.x - lhs.x*rhs.y;
        return complex<T>(r/d,i/d);
};

/**
 * Dividing by a number
 */
__host__ __device__
friend complex<T> operator/(const complex<T>& lhs, T d){
        return complex<T>(lhs.x/d,lhs.y/d);
};

};


/**
 * Prints the real and imaginary parts of the complex number
 */
template <class T>
__host__ __device__
void print(complex<T> in){
        printf("Real:\t%.15g\tImag:\t%.15g\n", in.real(), in.imag());
};


/**
 * Gets the absulute value of a complex number
 * @param  in - Complex number
 * @return    - Type of complex
 */
template <class T>
__host__ __device__
T abs(complex<T> in){
        return sqrt(in.real()*in.real() + in.imag()*in.imag());
};

/**
 * Gets the complex conjugate
 * @param  in - Complex number
 * @return    - Complex number
 */
template <class T>
__host__ __device__
complex<T> conj(complex<T> in){
        return complex<T>(in.real(), (-1)*in.imag());
};




} //End of namespace bach
#endif
