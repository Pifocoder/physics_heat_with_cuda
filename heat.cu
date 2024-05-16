/*
 * Based on CSC materials from:
 * 
 * https://github.com/csc-training/openacc/tree/master/exercises/heat
 *
 */
#include <cmath>
#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "pngwriter.h"

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define BLOCK_SIZE_Z 16
/* Convert 2D index layout to unrolled 1D layout
 *
 * \param[in] i      Row index
 * \param[in] j      Column index
 * \param[in] width  The width of the area
 * 
 * \returns An index in the unrolled 1D array.
 */
__host__ __device__ int getIndex(const int i, const int j, const int k,  const int width, const int deep)
{
    return (i*width + j)*deep + k;
}

__global__ void heat_kernel(int nx, int ny, int nz, double* d_Un, double* d_Unp1, double aTimesDt, double dx2, double dy2, double dz2)
{
    // Going through the entire area
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i > 0 && i < nx-1)
    {
        int j = threadIdx.y + blockIdx.y*blockDim.y;
        if (j > 0 && j < ny-1)
        {
            int k = threadIdx.z + blockIdx.z*blockDim.z;
            if (k > 0 && k < nz-1) {
                const int index = getIndex(i, j, k, ny, nz);
                double uij = d_Un[index];
                double uim1jk = d_Un[getIndex(i-1, j, k, ny, nz)];
                double uijm1k = d_Un[getIndex(i, j-1, k, ny, nz)];
                double uijkm1 = d_Un[getIndex(i, j, k-1, ny, nz)];
                
                double uip1jk = d_Un[getIndex(i+1, j, k, ny, nz)];
                double uijp1k = d_Un[getIndex(i, j+1, k, ny, nz)];
                double uijkp1 = d_Un[getIndex(i, j, k+1, ny, nz)];
                
                // Explicit scheme
                d_Unp1[index] = uij + aTimesDt * ( (uim1jk - 2.0*uij + uip1jk)/dx2 + (uijm1k - 2.0*uij + uijp1k)/dy2+ (uijkm1 - 2.0*uij + uijkp1)/dz2);
            }
        }
    }
}


int main()
{
    const int nx = 200;   // Width of the area
    const int ny = 200;   // Height of the area
    const int nz = 200;   // Depth of the area

    double a;     // Diffusion constant
    std::cout << "Enter the diffusion constant (a): ";
    std::cin >> a;

    const double dx = 0.01;   // Horizontal grid spacing 
    const double dy = 0.01;   // Vertical grid spacing
    const double dz = 0.01;

    const double dx2 = dx*dx;
    const double dy2 = dy*dy;
    const double dz2 = dz*dz;

    const double dt = dx2 * dy2 * dz2/ (2.0 * a * (dx2 + dy2 + dz2)); // Largest stable time step
    const int numSteps = 500000;                                       // Number of time steps
    const int outputEvery = 1000;                                    // How frequently to write output image

    int numElements = nx*ny*nz;

    // Allocate two sets of data for current and next timesteps
    double* h_Un   = (double*)calloc(numElements, sizeof(double));
    double* h_Unp1 = (double*)calloc(numElements, sizeof(double));

    double* d_Un;
    double* d_Unp1;

    cudaMalloc((void**)&d_Un, numElements*sizeof(double));
    cudaMalloc((void**)&d_Unp1, numElements*sizeof(double));

    
    // Initializing the data with a pattern of disk of radius of 1/6 of the width
    double object_x;
    std::cout << "Enter the width of the object ";
    std::cin >> object_x;
    double object_y;
    std::cout << "Enter the height of the object ";
    std::cin >> object_y;
    double object_z;
    std::cout << "Enter the deep of the object ";
    std::cin >> object_z;
    
    double center_x = nx/2;
    double center_y = ny/2;
    double center_z = nz/2;

    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            for (int k = 0; k < nz; ++k) {
                int index = getIndex(i, j, k, ny, nz);
                // Distance of point i, j from the origin
                if ((abs(i - center_x) <= object_x / 2) && (abs(j - center_y) <= object_y / 2) && (abs(k - center_z) <= object_z / 2)) {
                    h_Un[index] = 10.0;
                } else {
                    h_Un[index] = 5.0;
                }
            }
        }
    }

    // Fill in the data on the next step to ensure that the boundaries are identical.
    memcpy(h_Unp1, h_Un, numElements*sizeof(double));

    cudaMemcpy(d_Un, h_Un, numElements*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Unp1, d_Un, numElements*sizeof(double), cudaMemcpyDeviceToDevice);

    // Timing
    clock_t start = clock();

    dim3 numBlocks(nx/BLOCK_SIZE_X + 1, ny/BLOCK_SIZE_Y + 1, nz/BLOCK_SIZE_Z + 1);
    dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);

    // Main loop
    for (int n = 0; n <= numSteps; n++)
    {
        heat_kernel<<<numBlocks, threadsPerBlock>>>(nx, ny, nz, d_Un, d_Unp1, a*dt, dx2, dy2, dz2);
        // Write the output if needed
        if (n % outputEvery == 0)
        {
            cudaMemcpy(h_Un, d_Un, numElements*sizeof(double), cudaMemcpyDeviceToHost);

            int sum_temp_in = 0;
            int number_in = 0;
            int sum_temp_out = 0;
            int number_out = 0;
            double res = -10000000000;
            for (int i = 0; i < nx; i++)
            {
                for (int j = 0; j < ny; j++)
                {
                    for (int k = 0; k < nz; ++k) {
                        int index = getIndex(i, j,k, ny, nz);
                        // Distance of point i, j from the origin
                        if ((abs(i - center_x) <= object_x / 2) && (abs(j - center_y) <= object_y / 2) && (abs(k - center_z) <= object_z / 2)) {
                            sum_temp_in += h_Un[index];
                            ++number_in;
                        } else {
                            sum_temp_out += h_Un[index];
                            ++number_out;
                        }

                        double uij = h_Un[index];
                        double uim1jk = h_Un[getIndex(i-1, j, k, ny, nz)];
                        double uijm1k = h_Un[getIndex(i, j-1, k, ny, nz)];
                        double uijkm1 = h_Un[getIndex(i, j, k-1, ny, nz)];
                        
                        double uip1jk = h_Un[getIndex(i+1, j, k, ny, nz)];
                        double uijp1k = h_Un[getIndex(i, j+1, k, ny, nz)];
                        double uijkp1 = h_Un[getIndex(i, j, k+1, ny, nz)];
                        
                        // Explicit scheme
                        res = std::max(a*dt * ( (uim1jk - 2.0*uij + uip1jk)/dx2 + (uijm1k - 2.0*uij + uijp1k)/dy2 + (uijkm1 - 2.0*uij + uijkp1)/dz2), res);
                            }
                }
            }
            std::cout << res << std::endl;
            std::cout << "Mean temperature in the start zone" << sum_temp_in << " " << number_in << ", out: " << sum_temp_out << " " <<  number_out << std::endl;
            //save_stats(h_Un, nx, ny, nz, object_x, object_y, object_z, filename, 'c');
        }
        // Swapping the pointers for the next timestep
        std::swap(d_Un, d_Unp1);
    }

    // Timing
    clock_t finish = clock();
    printf("It took %f seconds\n", (double)(finish - start) / CLOCKS_PER_SEC);

    // Release the memory
    cudaFree(d_Un);
    cudaFree(d_Unp1);
    free(h_Un);
    free(h_Unp1);
    
    return 0;
}