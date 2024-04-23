#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <assert.h>

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

#define SOFTENING 1e-9f

/*
 * Each body contains x, y, and z coordinate positions,
 * as well as velocities in the x, y, and z directions.
 */

typedef struct { float x, y, z, vx, vy, vz; } Body;

/*
 * Calculate the gravitational impact of all bodies in the system on all others.
 */
 __global__ void bodyForce(Body *p, float dt, int n) {
  // int index = threadIdx.x;
  // int stride = blockDim.x;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  
  for (int i = index; i < n; i += stride) {
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = rsqrtf(distSqr); //reciprical of the square root
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
    }

    p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;

    // if(threadIdx.x == 255 && blockIdx.x == 15)
    // {
    //   printf("in kernel!\n");
    // }
  }
}


int main(const int argc, const char** argv) {


  int nBodies = 2<<11;
  if (argc > 1) nBodies = 2<<atoi(argv[1]);

  const float dt = 0.01f; // Time step
  const int nIters = 10;  // Simulation iterations

  int bytes = nBodies * sizeof(Body);
  float *buf;

  cudaError_t mallocErr, syncErr, asyncErr;
  mallocErr = cudaMallocManaged(&buf, bytes);
  checkCuda(mallocErr);

  Body *p = (Body*)buf; // cast to array of Body structs


  for (int iter = 0; iter < nIters; iter++) {

    int numThreads = 256;
    // int numBlocks = 1;
    int blockSize = 256;
    int numBlocks = (nBodies + blockSize - 1) / blockSize;
    // printf("numBlocks%i\n", numBlocks); // 16
    bodyForce<<<numBlocks, numThreads>>>(p, dt, nBodies); 
    syncErr = cudaGetLastError();
    asyncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) printf("Sync Error: %s\n", cudaGetErrorString(syncErr));
    if (asyncErr != cudaSuccess) printf("Async Error: %s\n", cudaGetErrorString(asyncErr));

    for (int i = 0 ; i < nBodies; i++) { // integrate position
      p[i].x += p[i].vx*dt;
      p[i].y += p[i].vy*dt;
      p[i].z += p[i].vz*dt;
    }

  }

  cudaFree(buf);	

  printf("finished\n");
}