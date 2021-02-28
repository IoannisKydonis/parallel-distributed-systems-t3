#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

float* gaussianDistribution(int range, float sigma) {
    float* distribution = (float*)malloc(range * sizeof(float));
    float mu = (float)(range - 1) / 2.0f;
    for (int i = 0; i < range; i++) {
        distribution[i] = exp(-pow(((float)i - mu) / sigma, 2.0f) / 2.0f) / (sigma * sqrt(2.0f * (float)M_PI));
    }
    return distribution;
}

int positiveElseZero(int n) {
    return n < 0 ? 0 : n;
}

int actualElseMax(int n, int patchSize, int max) {
    return n + patchSize > max ? max : n;
}

void nonLocalMeans(float *f, float *fh, int imageWidth, int imageHeight, int patchWidth, int patchHeight, float patchSigma, float gaussianCoreSigma) {
    float *gaussianCore = gaussianDistribution(patchWidth * patchHeight, gaussianCoreSigma);
    for (int i1 = 0; i1 < imageHeight; i1++) {
        for (int i2 = 0; i2 < imageWidth; i2++) {
            float z = 0;
            float w = 0;
            for (int j1 = 0; j1 < imageHeight; j1++) {
                for (int j2 = 0; j2 < imageWidth; j2++) {
                    // Calculate patch coordinates
                    int p1Height = positiveElseZero(actualElseMax(i1 - patchHeight / 2, patchHeight, imageHeight));
                    int p1Width = positiveElseZero(actualElseMax(i2 - patchWidth / 2, patchWidth, imageWidth));
                    int p2Height = positiveElseZero(actualElseMax(j1 - patchHeight / 2, patchHeight, imageHeight));
                    int p2Width = positiveElseZero(actualElseMax(j2 - patchWidth / 2, patchWidth, imageWidth));
                    int p1 = p1Height * imageWidth + p1Width;
                    int p2 = p2Height * imageWidth + p2Width;
                    float tempw = 0.0f;
                    for (int i = 0; i < patchWidth * patchHeight; i++) {
                        int pos1 = p1 + i / patchWidth * imageWidth + i % patchWidth;
                        int pos2 = p2 + i / patchWidth * imageWidth + i % patchWidth;
                        float fPos1 = f[pos1];
                        float fPos2 = f[pos2];
                        tempw += powf(fPos1 - fPos2, 2.0f) * gaussianCore[i];
                    }
                    tempw /= (patchSigma * patchSigma);
                    tempw = expf(-tempw);
                    float ff = f[j1 * imageWidth + j2];
                    w += tempw * ff;
                    z += tempw;
                }
            }
            fh[i1 * imageWidth + i2] = w / z;
        }
    }
    free(gaussianCore);
}

int main(int argc, char *argv[]) {
    float patchSigma = 0.05f;
    float coreSigma = 1.2f;

    if (argc < 6) {
        fprintf(stderr, "Usage: %s [imagePath] [imageWidth] [imageHeight] [patchWidth] [patchHeight]\n", argv[0]);
        return EXIT_FAILURE;
    }
    char *imagePath = argv[1];
    int imageWidth = atoi(argv[2]);
    int imageHeight = atoi(argv[3]);
    int patchWidth = atoi(argv[4]);
    int patchHeight = atoi(argv[5]);

    float* data = (float*)malloc(imageWidth * imageHeight * sizeof(float));
    float* fh = (float*)malloc(imageWidth * imageHeight * sizeof(float));

    FILE *fp = fopen(imagePath, "r");
    for (int i = 0; i < imageWidth * imageHeight - 1; i++) {
        fscanf(fp, "%f, ", data + i);
    }
    fscanf(fp, "%f", data + imageWidth * imageHeight - 1);
    fclose(fp);

    
    struct timespec ts_start;
    struct timespec ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    nonLocalMeans(data, fh, imageWidth, imageHeight, patchWidth, patchHeight, patchSigma, coreSigma);
    
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double tt = (ts_end.tv_sec - ts_start.tv_sec) + (ts_end.tv_nsec - ts_start.tv_nsec) / 1000000000.0;

    char* filename = (char*)malloc(20 * sizeof(char));
    sprintf(filename, "v0_%04d_%04d_%02d_%02d.txt", imageWidth, imageHeight, patchWidth, patchHeight);

    FILE* f = fopen(filename, "wb");
    fprintf(f, "Image Width: %d\n", imageWidth);
    fprintf(f, "Image Height: %d\n", imageHeight);
    fprintf(f, "Patch Width: %d\n", patchWidth);
    fprintf(f, "Patch Height: %d\n", patchHeight);
    fprintf(f, "Time: %lf sec\n", tt);
    fclose(f);

    for (int i = 0; i < imageWidth; i++) {
        for (int j = 0; j < imageHeight; j++) {
            printf("%10.8f ", fh[i * imageWidth + j]);
        }
        printf("\n");
    }

    free(fh);
    return EXIT_SUCCESS;
}