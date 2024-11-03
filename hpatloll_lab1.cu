#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cuda.h>

// CUDA kernel for tiled matrix multiplication
__global__ void TiledMatrixMultiplyKernel(float* A, float* B, float* C, int numARows, int numACols, int numBCols) {
    __shared__ float tileA[32][32];
    __shared__ float tileB[32][32];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;

    for (int m = 0; m < (numACols + 31) / 32; ++m) {
        if (row < numARows && m * 32 + threadIdx.x < numACols) {
            tileA[threadIdx.y][threadIdx.x] = A[row * numACols + m * 32 + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < numBCols && m * 32 + threadIdx.y < numACols) {
            tileB[threadIdx.y][threadIdx.x] = B[(m * 32 + threadIdx.y) * numBCols + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int e = 0; e < 32; ++e) {
            sum += tileA[threadIdx.y][e] * tileB[e][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < numARows && col < numBCols) {
        C[row * numBCols + col] = sum;
    }
}

// Host function for matrix multiplication on the CPU
void MatrixMultiplyCPU(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int numARows, int numACols, int numBCols) {
    for (int i = 0; i < numARows; ++i) {
        for (int j = 0; j < numBCols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < numACols; ++k) {
                sum += A[i * numACols + k] * B[k * numBCols + j];
            }
            C[i * numBCols + j] = sum;
        }
    }
}

// Function to verify the result
bool VerifyResults(const std::vector<float>& C_cpu, const std::vector<float>& C_gpu, int size) {
    const float epsilon = 1e-5;
    for (int i = 0; i < size; ++i) {
        if (fabs(C_cpu[i] - C_gpu[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

// Main matrix multiplication function
void MatrixMultiplication(int numARows, int numACols, int numBCols) {
    // Allocate host memory
    std::vector<float> h_A(numARows * numACols);
    std::vector<float> h_B(numACols * numBCols);
    std::vector<float> h_C_cpu(numARows * numBCols);
    std::vector<float> h_C_gpu(numARows * numBCols);

    // Initialize matrices with random values
    srand(time(0));
    for (int i = 0; i < numARows * numACols; ++i) h_A[i] = rand() % 100;
    for (int i = 0; i < numACols * numBCols; ++i) h_B[i] = rand() % 100;

    // Measure CPU time
    auto start_cpu = std::chrono::high_resolution_clock::now();
    MatrixMultiplyCPU(h_A, h_B, h_C_cpu, numARows, numACols, numBCols);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(end_cpu - start_cpu).count();
    std::cout << "CPU Matrix Multiplication Time: " << cpu_time << " seconds" << std::endl;

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, numARows * numACols * sizeof(float));
    cudaMalloc((void**)&d_B, numACols * numBCols * sizeof(float));
    cudaMalloc((void**)&d_C, numARows * numBCols * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_A, h_A.data(), numARows * numACols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), numACols * numBCols * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(32, 32);
    dim3 gridSize((numBCols + 31) / 32, (numARows + 31) / 32);

    // Measure GPU time
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu);
    TiledMatrixMultiplyKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, numARows, numACols, numBCols);
    cudaEventRecord(stop_gpu);

    // Wait for GPU to finish
    cudaEventSynchronize(stop_gpu);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);
    std::cout << "GPU Matrix Multiplication Time: " << gpu_time / 1000.0 << " seconds" << std::endl;

    // Copy result back to host
    cudaMemcpy(h_C_gpu.data(), d_C, numARows * numBCols * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify results
    std::cout << "Verifying Results..." << std::endl;
    if (VerifyResults(h_C_cpu, h_C_gpu, numARows * numBCols)) {
        std::cout << "Result Verification: PASSED (GPU output matches CPU reference)" << std::endl;
    } else {
        std::cout << "Result Verification: FAILED (GPU output does not match CPU reference)" << std::endl;
    }

    // Print performance metrics
    double speedup = cpu_time / (gpu_time / 1000.0);
    std::cout << "Speedup Achieved: " << speedup << "x faster than CPU" << std::endl;

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    std::cout << "Starting Tiled Matrix Multiplication Program..." << std::endl;

    // Define matrix sizes (example: 1024 x 512 and 512 x 2048)
    int numARows = 1024;
    int numACols = 512;
    int numBCols = 2048;

    MatrixMultiplication(numARows, numACols, numBCols);

    std::cout << "Program Completed Successfully." << std::endl;
    return 0;
}
