#include "common.cuh"

__global__ void sgemm_naive(int M, int N, int K, float alpha,
                            const float* A, const float* B,
                            float beta, float* C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < N && row < M) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

void launch_naive(int M, int N, int K, float alpha, const float* A,
                  const float* B, float beta, float* C) {
    dim3 block(32, 32);
    dim3 grid((N + 32 - 1) / 32, (M + 32 - 1) / 32);
    sgemm_naive<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

int main() {
    run_sgemm("naive", launch_naive, 100, 70, 50);
    run_sgemm("naive", launch_naive, 4096, 4096, 4096);
}