/*
 * Exercise 2: GPU Memory Allocation and Transfer Benchmark
 * =========================================================
 * Benchmarks the impact of different host and device memory allocation
 * strategies on H2D/D2H transfer times and kernel execution time.
 *
 * Usage:
 * ./mem_bench <host_type> <device_type> <size_mb>
 *
 * Arguments:
 * host_type   : 0 = pageable (malloc)
 * 1 = pinned (cudaMallocHost)
 * 2 = write-combining pinned (cudaHostAllocWriteCombined)
 * 3 = managed (cudaMallocManaged) [acts as host buffer]
 * 4 = mapped (cudaHostAllocMapped) [zero-copy, no explicit transfer]
 *
 * device_type : 0 = standard device memory (cudaMalloc)
 * 1 = managed (cudaMallocManaged) [accessed from device]
 *
 * size_mb     : size of the array in megabytes (e.g. 64, 256, 512)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

// ── Constants ────────────────────────────────────────────────────────────────
#define BLOCK_SIZE  256
#define WARMUP_REPS 5    
#define TIMED_REPS  20   

// Host memory type labels
static const char* HOST_LABELS[] = {
    "pageable      ",
    "pinned        ",
    "write-combine ",
    "managed       ",
    "mapped(no xfr)"
};

// Device memory type labels
static const char* DEV_LABELS[] = {
    "device  ",
    "managed "
};

// ── Kernel ───────────────────────────────────────────────────────────────────
__global__ void vector_scale(const float* __restrict__ in,
                                    float* out,
                                    int                 n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = in[gid] * 2.0f + 1.0f;
}

__global__ void vector_scale_inplace(float* buf, int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) buf[gid] = buf[gid] * 2.0f + 1.0f;
}

__global__ void warmup_kernel() {}

// ── Statistics helpers ───────────────────────────────────────────────────────
static float compute_mean(float* vals, int n)
{
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += vals[i];
    return s / n;
}

static int cmp_float(const void* a, const void* b)
{
    float fa = *(float*)a, fb = *(float*)b;
    return (fa > fb) - (fa < fb);
}

static float compute_median(float* vals, int n)
{
    float* tmp = (float*)malloc(n * sizeof(float));
    memcpy(tmp, vals, n * sizeof(float));
    qsort(tmp, n, sizeof(float), cmp_float);
    float med = (n % 2 == 0)
              ? (tmp[n/2 - 1] + tmp[n/2]) * 0.5f
              : tmp[n/2];
    free(tmp);
    return med;
}

static float compute_stddev(float* vals, int n, float mean)
{
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += (vals[i] - mean) * (vals[i] - mean);
    return sqrtf(s / n);
}

// ── CUDA error check ─────────────────────────────────────────────────────────
#define CHECK(call) \
    do { \
        cudaError_t e = (call); \
        if (e != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d — %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(1); \
        } \
    } while(0)

// ── Event timing helpers ─────────────────────────────────────────────────────
static float event_ms(cudaEvent_t a, cudaEvent_t b)
{
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    return ms;
}

// ── Host allocation ──────────────────────────────────────────────────────────
static float* alloc_host(int host_type, size_t bytes, float** mapped_dev_ptr)
{
    float* ptr = NULL;
    *mapped_dev_ptr = NULL;

    switch (host_type) {
        case 0: ptr = (float*)malloc(bytes); break;
        case 1: CHECK(cudaMallocHost(&ptr, bytes)); break;
        case 2: CHECK(cudaHostAlloc(&ptr, bytes, cudaHostAllocWriteCombined)); break;
        case 3: CHECK(cudaMallocManaged(&ptr, bytes)); break;
        case 4: 
            CHECK(cudaHostAlloc(&ptr, bytes, cudaHostAllocMapped));
            CHECK(cudaHostGetDevicePointer((void**)mapped_dev_ptr, ptr, 0));
            break;
    }
    return ptr;
}

static void free_host(int host_type, float* ptr)
{
    switch (host_type) {
        case 0: free(ptr);            break;
        case 1: cudaFreeHost(ptr);    break;
        case 2: cudaFreeHost(ptr);    break;
        case 3: cudaFree(ptr);        break;
        case 4: cudaFreeHost(ptr);    break;
    }
}

// ── Device allocation ─────────────────────────────────────────────────────────
static float* alloc_device(int dev_type, size_t bytes)
{
    float* ptr = NULL;
    switch (dev_type) {
        case 0: CHECK(cudaMalloc(&ptr, bytes));        break;
        case 1: CHECK(cudaMallocManaged(&ptr, bytes)); break;
    }
    return ptr;
}

// ── Print table row ───────────────────────────────────────────────────────────
static void print_row(const char* label, float* vals, int n, size_t bytes)
{
    float mean   = compute_mean(vals, n);
    float median = compute_median(vals, n);
    float stddev = compute_stddev(vals, n, mean);
    float bw     = (float)bytes / (mean * 1e6f);  // GB/s
    printf("  %-20s  mean=%8.3f ms  median=%8.3f ms  "
           "stddev=%7.3f ms  bw=%6.2f GB/s\n",
           label, mean, median, stddev, bw);
}

// ── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    if (argc != 4) {
        fprintf(stderr,
            "Usage: %s <host_type> <device_type> <size_mb>\n"
            "  host_type  : 0=pageable 1=pinned 2=write-combine 3=managed 4=mapped\n"
            "  device_type: 0=device   1=managed\n"
            "  size_mb    : array size in MB\n",
            argv[0]);
        return 1;
    }

    int   host_type = atoi(argv[1]);
    int   dev_type  = atoi(argv[2]);
    int   size_mb   = atoi(argv[3]);

    if (host_type < 0 || host_type > 4) {
        fprintf(stderr, "Error: host_type must be 0–4\n"); return 1; }
    if (dev_type < 0 || dev_type > 1) {
        fprintf(stderr, "Error: device_type must be 0 or 1\n"); return 1; }
    if (size_mb < 1) {
        fprintf(stderr, "Error: size_mb must be >= 1\n"); return 1; }

    if (host_type == 4 && dev_type == 1) {
        fprintf(stderr, "Error: mapped host memory is incompatible with managed device memory.\n");
        return 1;
    }

    size_t bytes = (size_t)size_mb * 1024 * 1024;
    int    n     = (int)(bytes / sizeof(float));
    int    grid  = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("=============================================================\n");
    printf("Device:       %s\n", prop.name);
    printf("Host memory:  %s (type %d)\n", HOST_LABELS[host_type], host_type);
    printf("Device mem:   %s (type %d)\n", DEV_LABELS[dev_type],   dev_type);
    printf("Array size:   %d MB  (%d floats)\n", size_mb, n);
    printf("Iterations:   %d timed  (+ %d discarded warmups)\n", TIMED_REPS, WARMUP_REPS);
    printf("=============================================================\n\n");

    // ── Timing arrays ────────────────────────────────────────────────────────
    float *t_alloc_h  = (float*)malloc(TIMED_REPS * sizeof(float));
    float *t_alloc_d  = (float*)malloc(TIMED_REPS * sizeof(float));
    float *t_h2d      = (float*)malloc(TIMED_REPS * sizeof(float));
    float *t_kernel   = (float*)malloc(TIMED_REPS * sizeof(float));
    float *t_d2h      = (float*)malloc(TIMED_REPS * sizeof(float));
    float *t_free_h   = (float*)malloc(TIMED_REPS * sizeof(float));
    float *t_free_d   = (float*)malloc(TIMED_REPS * sizeof(float));

    // ── CUDA events ───────────────────────────────────────────────────────────
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    // ── Hardware warmup (CUDA context initialization) ─────────────────────────
    cudaFree(0);
    warmup_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    int total_reps = WARMUP_REPS + TIMED_REPS;
    
    // Timer for overall benchmark execution
    struct timespec bench_start, bench_end;
    clock_gettime(CLOCK_MONOTONIC, &bench_start);

    // ── Main benchmark loop ───────────────────────────────────────────────────
    for (int rep = 0; rep < total_reps; rep++) {
        int timed = (rep >= WARMUP_REPS);
        int ti    = rep - WARMUP_REPS;  // index into timing arrays

        struct timespec wall0, wall1;

        // ── Host allocation ──────────────────────────────────────────────────
        float* mapped_dev_ptr = NULL;

        clock_gettime(CLOCK_MONOTONIC, &wall0);
        float* h_buf = alloc_host(host_type, bytes, &mapped_dev_ptr);
        clock_gettime(CLOCK_MONOTONIC, &wall1);

        if (timed)
            t_alloc_h[ti] = (wall1.tv_sec  - wall0.tv_sec)  * 1e3f
                          + (wall1.tv_nsec - wall0.tv_nsec) * 1e-6f;

        for (int i = 0; i < n; i++) h_buf[i] = (float)(i % 1000) * 0.001f;

        // ── Device allocation ────────────────────────────────────────────────
        clock_gettime(CLOCK_MONOTONIC, &wall0);
        float* d_in  = alloc_device(dev_type, bytes);
        float* d_out = alloc_device(dev_type, bytes);
        clock_gettime(CLOCK_MONOTONIC, &wall1);

        if (timed)
            t_alloc_d[ti] = (wall1.tv_sec  - wall0.tv_sec)  * 1e3f
                          + (wall1.tv_nsec - wall0.tv_nsec) * 1e-6f;

        // ── H2D transfer ─────────────────────────────────────────────────────
        if (host_type == 4) {
            if (timed) t_h2d[ti] = 0.0f;
        } else {
            cudaEventRecord(e0);
            CHECK(cudaMemcpy(d_in, h_buf, bytes, cudaMemcpyHostToDevice));
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            if (timed) t_h2d[ti] = event_ms(e0, e1);
        }

        // ── Kernel ───────────────────────────────────────────────────────────
        cudaEventRecord(e0);
        if (host_type == 4) {
            vector_scale_inplace<<<grid, BLOCK_SIZE>>>(mapped_dev_ptr, n);
        } else {
            vector_scale<<<grid, BLOCK_SIZE>>>(d_in, d_out, n);
        }
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        if (timed) t_kernel[ti] = event_ms(e0, e1);

        // ── D2H transfer ─────────────────────────────────────────────────────
        if (host_type == 4) {
            if (timed) t_d2h[ti] = 0.0f;
        } else {
            cudaEventRecord(e0);
            CHECK(cudaMemcpy(h_buf, d_out, bytes, cudaMemcpyDeviceToHost));
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            if (timed) t_d2h[ti] = event_ms(e0, e1);
        }

        // ── Free device ──────────────────────────────────────────────────────
        clock_gettime(CLOCK_MONOTONIC, &wall0);
        cudaFree(d_in);
        cudaFree(d_out);
        clock_gettime(CLOCK_MONOTONIC, &wall1);
        if (timed)
            t_free_d[ti] = (wall1.tv_sec  - wall0.tv_sec)  * 1e3f
                         + (wall1.tv_nsec - wall0.tv_nsec) * 1e-6f;

        // ── Free host ────────────────────────────────────────────────────────
        clock_gettime(CLOCK_MONOTONIC, &wall0);
        free_host(host_type, h_buf);
        clock_gettime(CLOCK_MONOTONIC, &wall1);
        if (timed)
            t_free_h[ti] = (wall1.tv_sec  - wall0.tv_sec)  * 1e3f
                         + (wall1.tv_nsec - wall0.tv_nsec) * 1e-6f;
    }

    clock_gettime(CLOCK_MONOTONIC, &bench_end);
    float total_elapsed_ms = (bench_end.tv_sec  - bench_start.tv_sec)  * 1e3f
                           + (bench_end.tv_nsec - bench_start.tv_nsec) * 1e-6f;

    // ── Print results ─────────────────────────────────────────────────────────
    printf("Results (all times in ms, bandwidth = array_size / mean_time):\n\n");

    print_row("host alloc",   t_alloc_h, TIMED_REPS, bytes);
    print_row("device alloc", t_alloc_d, TIMED_REPS, bytes);

    if (host_type == 4) {
        printf("  %-20s  N/A (zero-copy: no explicit H2D transfer)\n", "H2D transfer");
        printf("  %-20s  N/A (zero-copy: no explicit D2H transfer)\n", "D2H transfer");
    } else {
        print_row("H2D transfer", t_h2d,  TIMED_REPS, bytes);
        print_row("D2H transfer", t_d2h,  TIMED_REPS, bytes);
    }

    print_row("kernel",       t_kernel,  TIMED_REPS, bytes);
    print_row("device free",  t_free_d,  TIMED_REPS, bytes);
    print_row("host free",    t_free_h,  TIMED_REPS, bytes);

    printf("-------------------------------------------------------------\n");
    printf("  Overall Elapsed Time for %d Timed Iterations: %.2f ms\n", TIMED_REPS, total_elapsed_ms);
    printf("  Average Time per Full Iteration (Timed):      %.2f ms\n", total_elapsed_ms / TIMED_REPS);
    printf("=============================================================\n");

    // ── Cleanup ───────────────────────────────────────────────────────────────
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    free(t_alloc_h); free(t_alloc_d);
    free(t_h2d);     free(t_d2h);
    free(t_kernel);  free(t_free_h); free(t_free_d);

    return 0;
}