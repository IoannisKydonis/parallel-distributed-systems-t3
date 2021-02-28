#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gputimer.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int sharedMemSize = 16384 / sizeof(float);

float* gaussianDistribution(int range, float sigma) {
    float* distribution = (float*)malloc(range * sizeof(float));
    float mu = (float)(range - 1) / 2.0f;
    for (int i = 0; i < range; i++) {
        distribution[i] = expf(-powf(((float)i - mu) / sigma, 2.0f) / 2.0f) / (sigma * sqrtf(2.0f * (float)M_PI));
    }
    return distribution;
}

__device__ int positiveElseZero(int n) {
    return n < 0 ? 0 : n;
}

__device__ int actualElseMax(int n, int patchSize, int max) {
    return n + patchSize > max ? max : n;
}

__global__ void nonLocalMeans(float *f, float *fh, int imageWidth, int imageHeight, int patchWidth, int patchHeight, float patchSigma, float *gaussianCore) {
    int i1 = blockIdx.y * blockDim.y + threadIdx.y;
    int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i1 >= imageWidth || i2 >= imageHeight)
        return;

    // Fill shared memory
    extern __shared__ float ss[];
    int sharedMemSize = 16384 / sizeof(float);
    int limit = imageWidth * imageHeight < sharedMemSize ? imageWidth * imageHeight : sharedMemSize;
    for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < limit; i += blockDim.x * blockDim.y) {
        ss[i] = f[i];
    }

    __syncthreads();

    float z = 0;
    float w = 0;
    int p1Height = positiveElseZero(actualElseMax(i1 - patchHeight / 2, patchHeight, imageHeight));
    int p1Width = positiveElseZero(actualElseMax(i2 - patchWidth / 2, patchWidth, imageWidth));
    int p1 = p1Height * imageWidth + p1Width;
    for (int j1 = 0; j1 < imageHeight; j1++) {
        for (int j2 = 0; j2 < imageWidth; j2++) {
            int p2Height = positiveElseZero(actualElseMax(j1 - patchHeight / 2, patchHeight, imageHeight));
            int p2Width = positiveElseZero(actualElseMax(j2 - patchWidth / 2, patchWidth, imageWidth));
            int p2 = p2Height * imageWidth + p2Width;
            float tempw = 0.0f;
            for (int i = 0; i < patchWidth * patchHeight; i++) {
                int pos1 = p1 + i / patchWidth * imageWidth + i % patchWidth;
                int pos2 = p2 + i / patchWidth * imageWidth + i % patchWidth;
                float fPos1 = pos1 < limit ? ss[pos1] : f[pos1];
                float fPos2 = pos2 < limit ? ss[pos2] : f[pos2];
                tempw += powf(fPos1 - fPos2, 2.0f) * gaussianCore[i];
            }
            tempw /= (patchSigma * patchSigma);
            tempw = expf(-tempw);
            float ff = j1 * imageWidth + j2 < limit ? ss[j1 * imageWidth + j2] : f[j1 * imageWidth + j2];
            w += tempw * ff;
            z += tempw;
        }
    }
    fh[i1 * imageWidth + i2] = w / z;
}

int main(int argc, char* argv[]) {
    float patchSigma = 0.05f;
    float coreSigma = 1.2f;

    if (argc < 6) {
        fprintf(stderr, "Usage: %s [imagePath] [imageWidth] [imageHeight] [patchWidth] [patchHeight]\n", argv[0]);
        return EXIT_FAILURE;
    }
    char* imagePath = argv[1];
    int imageWidth = atoi(argv[2]);
    int imageHeight = atoi(argv[3]);
    int patchWidth = atoi(argv[4]);
    int patchHeight = atoi(argv[5]);

    float* data = (float*)malloc(imageWidth * imageHeight * sizeof(float));

    FILE* fp = fopen(imagePath, "r");
    for (int i = 0; i < imageWidth * imageHeight - 1; i++) {
        fscanf(fp, "%f, ", data + i);
    }
    fscanf(fp, "%f", data + imageWidth * imageHeight - 1);
    fclose(fp);

    float *gaussianCore = gaussianDistribution(patchWidth * patchHeight, coreSigma);
    float *dgaussianCore;
    cudaMalloc(&dgaussianCore, patchWidth * patchHeight * sizeof(float));
    cudaMemcpy(dgaussianCore, gaussianCore, patchWidth * patchHeight * sizeof(float), cudaMemcpyHostToDevice);

    float *df;
    cudaMalloc(&df, imageWidth * imageHeight * sizeof(float));
    cudaMemcpy(df, data, imageWidth * imageHeight * sizeof(float), cudaMemcpyHostToDevice);
    float *dfh;
    cudaMalloc(&dfh, imageWidth * imageHeight * sizeof(float));

    dim3 dimBlock(16, 16);
    dim3 dimGrid((imageWidth + dimBlock.x - 1) / dimBlock.x, (imageHeight + dimBlock.y - 1) / dimBlock.y);
    int shareMemChosenSize = imageWidth * imageHeight < sharedMemSize ? imageWidth * imageHeight : sharedMemSize;

    GpuTimer t;
    t.Start();
    nonLocalMeans <<<dimGrid, dimBlock, shareMemChosenSize * sizeof(float)>>> (df, dfh, imageWidth, imageHeight, patchWidth, patchHeight, patchSigma, dgaussianCore);
    t.Stop();
    double tt = t.Elapsed() / 1000;

    float *fh = (float *)malloc(imageWidth * imageHeight * sizeof(float));
    cudaMemcpy(fh, dfh, imageWidth * imageHeight * sizeof(float), cudaMemcpyDeviceToHost);

    char* filename = (char*)malloc(20 * sizeof(char));
    sprintf(filename, "v2_%04d_%04d_%02d_%02d.txt", imageWidth, imageHeight, patchWidth, patchHeight);

    FILE* f = fopen(filename, "wb");
    fprintf(f, "Image Width: %d\n", imageWidth);
    fprintf(f, "Image Height: %d\n", imageHeight);
    fprintf(f, "Patch Width: %d\n", patchWidth);
    fprintf(f, "Patch Height: %d\n", patchHeight);
    fprintf(f, "Time: %lf sec\n", tt);
    fclose(f);

    for (int i = 0; i < imageHeight; i++) {
        for (int j = 0; j < imageWidth; j++) {
            printf("%10.8f ", fh[i * imageWidth + j]);
        }
        printf("\n");
    }

    cudaFree(dgaussianCore);
    cudaFree(df);
    cudaFree(dfh);
    free(fh);
    return EXIT_SUCCESS;
}