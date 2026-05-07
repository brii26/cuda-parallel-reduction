#include <cstdio>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>

#include "utils.h"
#include "data_gen.h"
#include "cpu_reduction.h"
#include "gpu_reduction.cuh"

static constexpr int NUM_RUNS         = 5;
static constexpr float REL_ERR_LIMIT  = 1e-2f;

// Benchmark helpers
struct BenchResult {
    double cpu_mean_ms, cpu_std_ms;
    double h2d_mean_ms, kernel_mean_ms, d2h_mean_ms;
    double gpu_mean_ms, gpu_std_ms;
    double speedup;
    bool   passed;
};

static BenchResult run_benchmark(int n) {
    const auto data = generate_data(n);

    // warm-up, not recorded
    {
        TimingResult tw;
        cpu_reduce(data.data(), n);
        gpu_reduce(data.data(), n, tw);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<double> cpu_ms(NUM_RUNS), gpu_ms(NUM_RUNS);
    std::vector<double> h2d_ms(NUM_RUNS), ker_ms(NUM_RUNS), d2h_ms(NUM_RUNS);
    float cpu_result = 0.f, gpu_result = 0.f;

    for (int r = 0; r < NUM_RUNS; ++r) {
        using Clock = std::chrono::high_resolution_clock;

        auto t0  = Clock::now();
        cpu_result = cpu_reduce(data.data(), n);
        auto t1  = Clock::now();
        cpu_ms[r] = std::chrono::duration<double, std::milli>(t1 - t0).count();

        TimingResult tr;
        gpu_result  = gpu_reduce(data.data(), n, tr);
        CUDA_CHECK(cudaDeviceSynchronize());
        gpu_ms[r] = tr.total_ms;
        h2d_ms[r] = tr.h2d_ms;
        ker_ms[r] = tr.kernel_ms;
        d2h_ms[r] = tr.d2h_ms;
    }

    const double cm = vec_mean(cpu_ms);
    const double gm = vec_mean(gpu_ms);

    BenchResult res;
    res.cpu_mean_ms   = cm;
    res.cpu_std_ms    = vec_stddev(cpu_ms, cm);
    res.h2d_mean_ms   = vec_mean(h2d_ms);
    res.kernel_mean_ms= vec_mean(ker_ms);
    res.d2h_mean_ms   = vec_mean(d2h_ms);
    res.gpu_mean_ms   = gm;
    res.gpu_std_ms    = vec_stddev(gpu_ms, gm);
    res.speedup       = cm / gm;
    res.passed        = fabsf(cpu_result - gpu_result) / fabsf(cpu_result) < REL_ERR_LIMIT;
    return res;
}

static void print_result(int n, const BenchResult& r) {
    printf("n = %10d\n", n);
    printf("  CPU  : %8.3f ± %.3f ms\n", r.cpu_mean_ms, r.cpu_std_ms);
    printf("  GPU  : %8.3f ± %.3f ms  (speedup %.2fx)  [%s]\n",
           r.gpu_mean_ms, r.gpu_std_ms, r.speedup, r.passed ? "PASS" : "FAIL");
    printf("  ├ H2D   : %.3f ms\n", r.h2d_mean_ms);
    printf("  ├ Kernel: %.3f ms\n", r.kernel_mean_ms);
    printf("  └ D2H   : %.3f ms\n", r.d2h_mean_ms);
}

static void write_csv_row(FILE* f, int n, const BenchResult& r) {
    fprintf(f, "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f\n",
            n,
            r.cpu_mean_ms, r.cpu_std_ms,
            r.h2d_mean_ms, r.kernel_mean_ms, r.d2h_mean_ms,
            r.gpu_mean_ms, r.gpu_std_ms,
            r.speedup);
}

// Entry point
int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device : %s  |  %d SMs  |  %zu KB shared/block  |  CUDA %d.%d\n\n",
           prop.name, prop.multiProcessorCount,
           prop.sharedMemPerBlock / 1024,
           prop.major, prop.minor);

    FILE* csv = fopen("results.csv", "w");
    if (!csv) { perror("fopen results.csv"); return EXIT_FAILURE; }
    fprintf(csv,
            "n,cpu_mean_ms,cpu_std_ms,h2d_mean_ms,kernel_mean_ms,"
            "d2h_mean_ms,gpu_total_mean_ms,gpu_total_std_ms,speedup\n");

    constexpr int sizes[] = {1'000, 100'000, 1'000'000, 100'000'000};
    for (int n : sizes) {
        const BenchResult r = run_benchmark(n);
        print_result(n, r);
        write_csv_row(csv, n, r);
        printf("\n");
    }

    fclose(csv);
    printf("Results saved to results.csv\n");
    return EXIT_SUCCESS;
}
