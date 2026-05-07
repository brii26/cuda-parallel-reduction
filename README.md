# cuda-parallel-reduction

Benchmarking serial CPU reduction vs parallel GPU reduction (CUDA) on float arrays of varying sizes.

## Structure

```
├── include/        # Headers
├── src/            # Source files (CPU, GPU, benchmark)
├── scripts/        # Python plotting script
└── Makefile
```

## Build

```bash
make ARCH=sm_86
```

## Run

```bash
./reduction    			          # outputs results.csv
python3 scripts/plot_results.py   # generates PNG plots
```

## Benchmark sizes

| n | Scale |
|---|---|
| 1,000 | Small |
| 100,000 | Medium |
| 1,000,000 | Large |
| 100,000,000 | Very large |

Each size runs 5 times (+ 1 warm-up). Reports mean/stddev for CPU and GPU, per-phase GPU timing (H2D, kernel, D2H), speedup, and numerical correctness.
