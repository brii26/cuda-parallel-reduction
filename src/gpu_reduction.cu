#include "gpu_reduction.cuh"
#include <vector>

static constexpr int BLOCK_SIZE = 256;

// ── Kernel ───────────────────────────────────────────────────────────────────
//
// Each block covers 2*BLOCK_SIZE consecutive elements.
// Threads beyond n contribute 0 (identity for addition).
// A single __syncthreads() barrier guards each halving step.

__global__ void reduce_kernel(const float* __restrict__ input,
                               float* __restrict__ output, int n)
{
    __shared__ float sdata[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x * 2 + tid;

    const float a = (idx              < n) ? input[idx]              : 0.f;
    const float b = (idx + BLOCK_SIZE < n) ? input[idx + BLOCK_SIZE] : 0.f;
    sdata[tid] = a + b;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// ── Public wrapper ───────────────────────────────────────────────────────────

float gpu_reduce(const float* h_input, int n, TimingResult& t) {
    const int numBlocks = (n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);

    float *d_input = nullptr, *d_partial = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input,   static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial, static_cast<size_t>(numBlocks) * sizeof(float)));

    cudaEvent_t ev[4];
    for (auto& e : ev) CUDA_CHECK(cudaEventCreate(&e));

    CUDA_CHECK(cudaEventRecord(ev[0]));
    CUDA_CHECK(cudaMemcpy(d_input, h_input,
                          static_cast<size_t>(n) * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(ev[1]));

    reduce_kernel<<<numBlocks, BLOCK_SIZE>>>(d_input, d_partial, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ev[2]));

    std::vector<float> h_partial(numBlocks);
    CUDA_CHECK(cudaMemcpy(h_partial.data(), d_partial,
                          static_cast<size_t>(numBlocks) * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(ev[3]));
    CUDA_CHECK(cudaEventSynchronize(ev[3]));

    CUDA_CHECK(cudaEventElapsedTime(&t.h2d_ms,    ev[0], ev[1]));
    CUDA_CHECK(cudaEventElapsedTime(&t.kernel_ms, ev[1], ev[2]));
    CUDA_CHECK(cudaEventElapsedTime(&t.d2h_ms,    ev[2], ev[3]));
    t.total_ms = t.h2d_ms + t.kernel_ms + t.d2h_ms;

    for (auto& e : ev) CUDA_CHECK(cudaEventDestroy(e));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_partial));

    // Final accumulation on host (CPU loop over per-block partial sums)
    float total = 0.f;
    for (int i = 0; i < numBlocks; ++i) total += h_partial[i];
    return total;
}
