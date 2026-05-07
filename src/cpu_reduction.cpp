#include "cpu_reduction.h"

float cpu_reduce(const float* arr, int n) {
    float sum = 0.f;
    for (int i = 0; i < n; ++i) sum += arr[i];
    return sum;
}
