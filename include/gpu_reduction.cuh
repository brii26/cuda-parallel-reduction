#pragma once

#include "utils.h"

/**
 * Reduce n floats on the GPU.
 *
 * Strategy:
 *   - Each block of BLOCK_SIZE threads reduces 2*BLOCK_SIZE elements using
 *     shared memory tree reduction (stride-halving).
 *   - Partial sums from each block are copied to host and summed with a CPU loop.
 *
 * @param h_input  Host pointer to the input array.
 * @param n        Number of elements.
 * @param t        Output: per-phase timing (H2D, kernel, D2H).
 * @return         Sum of all elements.
 */
float gpu_reduce(const float* h_input, int n, TimingResult& t);
