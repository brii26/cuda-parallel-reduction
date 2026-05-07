#pragma once

#include "utils.h"

// shared-memory tree reduction; partial sums accumulated on CPU
float gpu_reduce(const float* h_input, int n, TimingResult& t);
