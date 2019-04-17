//random number generator
//http://www.iquilezles.org/www/articles/sfrand/sfrand.htm
float sfrand( int *seed )
{
    float res;

    seed[0] *= 16807;

    *((unsigned int *) &res) = ( ((unsigned int)seed[0])>>9 ) | 0x40000000;

    return( res-3.0f );
}

#include<OpenCL\tensor.c>
#include<OpenCL\sort.c>