#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cub/cub.cuh>

#define PI 3.141592653589793
#define running_block_size 32
#define coul2Klevin 1.6710095663e+05




// __global__ defines the funciton that can be called from the host (CPU) and executed in the device (GPU)
__global__
void check_int(int n, int *x)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, j;
    for (j=0; j<n; j++)
    {
        for (i=index; i<n; i+=stride)
        {
            if (i==j)
            {
                // printf("index: %d\t%d\n", i, x[i]);
                printf("%d\n", x[i]);
            }
        }
    }
}

__global__
void check_int_custom(int n, int *x1, int *x2, int *x3, int *x4, int *x5, int *x6, int *x7, int *x8)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, j;
    for (j=0; j<n; j++)
    {
        for (i=index; i<n; i+=stride)
        {
            if (i==j)
            {
                // printf("index: %d\t%d\n", i, x[i]);
                printf("%d %d %d %d %d %d %d %d\n", x1[i], x2[i], x3[i], x4[i], x5[i], x6[i], x7[i], x8[i]);
            }
        }
    }
}

__global__
void check_double(int n, double *x)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, j;
    for (j=0; j<n; j++)
    {
        for (i=index; i<n; i+=stride)
        {
            if (i==j)
            {
                // printf("index: %d\t%lf\n", i, x[i]);
                printf("%lf\n", x[i]);
            }
        }
    }
}


__global__
void check_double_ini(int n, double *x)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, j;
    for (j=0; j<(n/3); j++)
    {
        for (i=index; i<n; i+=stride)
        {
            if (i==j)
            {
                // printf("index: %d\t%lf\n", i, x[i]);
                printf("%lf\t%lf\t%lf\n", x[3*i+0], x[3*i+1], x[3*i+2]);
            }
        }
    }
}



__global__
void check_double_sci(int n, double *x)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, j;
    for (j=0; j<n; j++)
    {
        for (i=index; i<n; i+=stride)
        {
            if (i==j)
            {
                // printf("index: %d\t%lf\n", i, x[i]);
                printf("%.5e\n", x[i]);
            }
        }
    }
}


__global__
void check_double_custom(int n, double *x1, double *x2)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, j;
    for (j=0; j<n; j++)
    {
        for (i=index; i<n; i+=stride)
        {
            if (i==j)
            {
                // printf("index: %d\t%lf\n", i, x[i]);
                // printf("%.3e\t%.3e\n", x1[i], x2[i]);
                printf("%lf %lf\n", x1[i], x2[i]);
            }
        }
    }
}

__global__
void check_double_custom2(int n, double *x1, double *x2, double *x3, double *x4, double *x5, double *x6)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, j;
    for (j=0; j<n; j++)
    {
        for (i=index; i<n; i+=stride)
        {
            if (i==j)
            {
                // printf("index: %d\t%lf\n", i, x[i]);
                printf("%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n", x1[i], x2[i], x3[i], x4[i], x5[i], x6[i]);
            }
        }
    }
}

__global__
void check_double_custom22(int n, double *x1, double *x2, double *x3, double *x4, double *x5, double *x6)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, j;
    for (j=0; j<n; j++)
    {
        for (i=index; i<n; i+=stride)
        {
            if (i==j)
            {
                // printf("index: %d\t%lf\n", i, x[i]);
                printf("%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n", x1[i], x2[i], x3[i], x4[i]-PI, x5[i], x6[i]);
            }
        }
    }
}

__global__
void check_double_custom3(int n, double *x1, double *x2, double *x3, double *x4, double *x5, double *x6)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, j;
    for (j=0; j<n; j++)
    {
        for (i=index; i<n; i+=stride)
        {
            if (i==j)
            {
                // printf("index: %d\t%lf\n", i, x[i]);
                printf("%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n", x1[i], x2[i], x3[i], x4[i]/2/PI*360, x5[i]/2/PI*360, x6[i]/2/PI*360);
            }
        }
    }
}


__global__
void check_double_custom4(int n, double *x1, double *x2, double *x3, double *x4, double *x5, double *x6, double *x7)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, j;
    for (j=0; j<n; j++)
    {
        for (i=index; i<n; i+=stride)
        {
            if (i==j)
            {
                // printf("index: %d\t%lf\n", i, x[i]);
                // printf("%.15e\t%.15e\t%.15e\t%.15e\t%.15e\t%.15e\t%.3e\n", x1[i], x2[i], x3[i], x4[i], x5[i], x6[i], x7[i]);
                // printf("%.10e\t%.10e\t%.10e\t%.10e\t%.10e\t%.10e\t%.3e\n", x1[i], x2[i], x3[i], x4[i], x5[i], x6[i], x7[i]);
                printf("%.5e\t%.5e\t%.5e\t%.5e\t%.5e\t%.5e\t%.3e\n", x1[i], x2[i], x3[i], x4[i], x5[i], x6[i], x7[i]);
            }
        }
    }
}


__global__
void check_double_custom5(int n, double *x1, double *x2, double *x3, double *x4, double *x5, double *x6, 
    double *y1, double *y2, double *y3, double *y4, double *y5, double *y6)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, j;
    for (j=0; j<n; j++)
    {
        for (i=index; i<n; i+=stride)
        {
            if (i==j)
            {
                // printf("index: %d\t%lf\n", i, x[i]);
                printf("%.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e\n", x1[i], x2[i], x3[i], x4[i], x5[i], x6[i], 
                    y1[i], y2[i], y3[i], y4[i], y5[i], y6[i]);
            }
        }
    }
}


__global__
void check_double_custom6(int n, double *x1, double *x2)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, j;
    for (j=0; j<n; j++)
    {
        for (i=index; i<n; i+=stride)
        {
            if (i==j)
            {
                // printf("index: %d\t%lf\n", i, x[i]);
                printf("%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\n", x1[i*3+0], x1[i*3+1], x1[i*3+2], 
                    x2[i*3+0], x2[i*3+1], x2[i*3+2]);
            }
        }
    }
}


__global__
void check_double_custom7(int n, double *x1, double *x2)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, j;
    for (j=0; j<n; j++)
    {
        for (i=index; i<n; i+=stride)
        {
            if (i==j)
            {
                // printf("index: %d\t%lf\n", i, x[i]);
                printf("%.3e %.3e %.3e %.3e\n", x1[i*2], 
                    x2[i*6+0], x2[i*6+1], x2[i*6+2]);
            }
        }
    }
}

__global__
void check_double_follow(double *x1, double *x2, double *x3)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, j;
    if (index==4)
    {
         printf("%lf\t%lf\t%lf\n", x1[3], x2[3], x3[3]);
    }
}



__global__
void check_double_temp(int n, double *x1, double *x2, double *x3)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, j;
    for (j=0; j<n; j++)
    {
        for (i=index; i<n; i+=stride)
        {
            if (i==j)
            {
                // printf("index: %d\t%lf\n", i, x[i]);
                printf("%lf\t%lf\t%lf\n", x1[i], x2[i], x3[i]);
            }
        }
    }
}

__global__
void check_double_angle1(int n, double *x1, double *x2, double *x3)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, j;
    for (i=index; i<n; i+=stride)
    {
        if ((i==399))
        {
            // printf("index: %d\t%lf\n", i, x[i]);
            printf("%d\t%lf\t%lf\t%lf\n", i, x1[i], x2[i], x3[i]);
        }
    }
}

__global__
void check_double_angle2(int n, double *x1, double *x2, double *x3)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, j;
    for (i=index; i<n; i+=stride)
    {
        if ((i==5))
        {
            // printf("index: %d\t%lf\n", i, x[i]);
            printf("%d\t%lf\t%lf\t%lf\n", i, x1[i], x2[i], x3[i]);
        }
    }
}

__global__
void check_double_angle3(int n, double *x1, double *x2, double *x3, double *x4)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, j;
    for (i=index; i<n; i+=stride)
    {
        if ((i==399))
        {
            // printf("index: %d\t%lf\n", i, x[i]);
            printf("%d\t%lf\t%lf\t%lf\t%lf\n", i, x1[i], x2[i], x3[i], x4[i]);
        }
    }
}





__global__
void check_key(int n, cub::KeyValuePair<int, double> *x)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, j;
    for (j=0; j<n; j++)
    {
        for (i=index; i<n; i+=stride)
        {
            if (i==j)
            {
                // printf("index: %d\t%lf\n", i, x[i]);
                printf("key: %d\tvalue: %lf\n", x[i].key, x[i].value);
            }
        }
    }
}

