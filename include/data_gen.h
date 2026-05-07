#pragma once

#include <vector>

/**
 * Generate n random floats in [0, 1) using a fixed seed.
 * Using the same seed guarantees identical data across serial and parallel runs.
 */
std::vector<float> generate_data(int n, unsigned seed = 42);
