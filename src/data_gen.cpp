#include "data_gen.h"
#include <cstdlib>

std::vector<float> generate_data(int n, unsigned seed) {
    srand(seed);
    std::vector<float> data(n);
    for (int i = 0; i < n; ++i)
        data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    return data;
}
