#pragma once

#include <cstdio>
#include <cmath>
#include <numeric>
#include <vector>
#include <cuda_runtime.h>

// CUDA error check
#define CUDA_CHECK(call) \
    do { \
        cudaError_t _err = (call); \
        if (_err != cudaSuccess) { \
            fprintf(stderr, "[CUDA Error] %s:%d  %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(_err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// per-phase GPU timings
struct TimingResult {
    float h2d_ms    = 0.f;
    float kernel_ms = 0.f;
    float d2h_ms    = 0.f;
    float total_ms  = 0.f;
};

// stats helpers
inline double vec_mean(const std::vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

inline double vec_stddev(const std::vector<double>& v, double m) {
    double acc = 0.0;
    for (double x : v) acc += (x - m) * (x - m);
    return std::sqrt(acc / static_cast<double>(v.size()));
}
