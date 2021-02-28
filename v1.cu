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

float* gaussianDistribution(int range, float sigma) {
    float* distribution = (float*)malloc(range * sizeof(float));
    float mu = (float)(range - 1) / 2.0f;
    for (int i = 0; i < range; i++) {
        distribution[i] = expf(-powf(((float)i - mu) / sigma, 2.0f) / 2.0f) / (sigma * sqrtf(2.0f * (float)M_PI));
    }
    return distribution;
}

__device__ float calculateWeight(float* f, int p1, int p2, int width, int height, float sigma, int imageWidth, float* gaussianCore) {
    float w = 0.0f;
    for (int i = 0; i < width * height; i++) {
        int pos1 = p1 + i / width * imageWidth + i % width;
        int pos2 = p2 + i / width * imageWidth + i % width;
        w += powf(f[pos1] - f[pos2], 2.0f) * gaussianCore[i];
    }
    w /= (sigma * sigma);
    w = expf(-w);
    return w;
}

__device__ int positiveElseZero(int n) {
    return n < 0 ? 0 : n;
}

__device__ int actualElseMax(int n, int patchSize, int max) {
    return n + patchSize > max ? max : n;
}

__device__ void getWeight(int i1, int i2, int j1, int j2, float *f, float *w, float *z, int imageWidth, int imageHeight, int patchWidth, int patchHeight, float patchSigma, float *gaussianCore) {
    // Calculate patch coordinates
    int p1Height = positiveElseZero(actualElseMax(i1 - patchHeight / 2, patchHeight, imageHeight));
    int p1Width = positiveElseZero(actualElseMax(i2 - patchWidth / 2, patchWidth, imageWidth));
    int p2Height = positiveElseZero(actualElseMax(j1 - patchHeight / 2, patchHeight, imageHeight));
    int p2Width = positiveElseZero(actualElseMax(j2 - patchWidth / 2, patchWidth, imageWidth));
    int p1 = p1Height * imageWidth + p1Width;
    int p2 = p2Height * imageWidth + p2Width;
    float tempw = calculateWeight(f, p1, p2, patchWidth, patchHeight, patchSigma, imageWidth, gaussianCore);
    w[0] += tempw * f[j1 * imageWidth + j2];
    z[0] += tempw;
}

__global__ void nonLocalMeans(float *f, float *fh, int imageWidth, int imageHeight, int patchWidth, int patchHeight, float patchSigma, float *gaussianCore) {
    int i1 = blockIdx.y * blockDim.y + threadIdx.y;
    int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i1 >= imageWidth || i2 >= imageHeight)
        return;
    
    float z = 0;
    float w = 0;
    for (int j1 = 0; j1 < imageHeight; j1++) {
        for (int j2 = 0; j2 < imageWidth; j2++) {
            getWeight(i1, i2, j1, j2, f, &w, &z, imageWidth, imageHeight, patchWidth, patchHeight, patchSigma, gaussianCore);
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

    GpuTimer t;
    t.Start();
    nonLocalMeans <<<dimGrid, dimBlock>>> (df, dfh, imageWidth, imageHeight, patchWidth, patchHeight, patchSigma, dgaussianCore);
    t.Stop();
    double tt = t.Elapsed() / 1000;

    float *fh = (float *)malloc(imageWidth * imageHeight * sizeof(float));
    cudaMemcpy(fh, dfh, imageWidth * imageHeight * sizeof(float), cudaMemcpyDeviceToHost);

    char* filename = (char*)malloc(20 * sizeof(char));
    sprintf(filename, "v1_%04d_%04d_%02d_%02d.txt", imageWidth, imageHeight, patchWidth, patchHeight);

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