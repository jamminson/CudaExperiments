#include "common.cuh"

#define TILE 32

__global__ void sgemm_tiled(int M, int N, int K, float alpha,
                            const float* A, const float* B,
                            float beta, float* C) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TILE) {
        As[ty][tx] = (row < M && k0 + tx < K) ? A[row * K + (k0 + tx)] : 0.0f;
        Bs[ty][tx] = (col < N && k0 + ty < K) ? B[(k0 + ty) * N + col] : 0.0f;

        __syncthreads();

        // Partial dot product
        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

void launch_tiled(int M, int N, int K, float alpha, const float* A,
                  const float* B, float beta, float* C) {
    dim3 block(32, 32);
    dim3 grid((N + 32 - 1) / 32, (M + 32 - 1) / 32);
    sgemm_tiled<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

int main() {
    run_sgemm("tiled", launch_tiled, 100, 70, 50);
    run_sgemm("tiled", launch_tiled, 4096, 4096, 4096);
}