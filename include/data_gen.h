#pragma once

#include <vector>

// uniform random floats in [0,1), fixed seed for reproducibility
std::vector<float> generate_data(int n, unsigned seed = 42);
