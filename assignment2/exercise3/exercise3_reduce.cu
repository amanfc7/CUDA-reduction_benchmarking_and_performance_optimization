/*
 * Exercise 3: Parallel Array Reduction (Student Template)
 * =========================================================
 *
 * Usage:
 * ./ex3_reduce <method> <size_mb>
 *
 * Arguments:
 * method  : 
 * 0 = CPU Reference (Sequential)
 * 1 = GPU Interleaved (Naive Shared Memory)
 * 2 = GPU Sequential  (Optimized Shared Memory)
 * 3 = GPU Warp-Shuffle (Register-level)
 *
 * size_mb : size of the array in megabytes (e.g., 64, 256)
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

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

// ── CPU Reference ────────────────────────────────────────────────────────────
float cpu_reduce(const float* h_in, int n) {
    double sum = 0.0; // using double to minimize floating point drift
    for (int i = 0; i < n; i++) {
        sum += h_in[i];
    }
    return (float)sum;
}

// ── Dummy Warmup Kernel ───────────────────────────────────────────────────────
__global__ void warmup_dummy(float* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        out[0] = 0.0f;
}

// ── Your Kernels Come Here ───────────────────────────────────────────────────

/*
 * Version A: Interleaved Tree Reduction
 * Stride grows each step (1 -> 2 -> 4 -> ... -> BLOCK_SIZE/2).
 * Active threads satisfy: tid % (2*stride) == 0
 * This interleaves active and idle threads within the same warp,
 * causing warp divergence and reduced SIMD utilisation.
 * Each block writes its partial sum to d_out[blockIdx.x].
 */
__global__ void kernel_interleaved(const float* __restrict__ d_in,
                                   float* __restrict__ d_out, int n)
{
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Load element; pad with 0 if out of bounds
    sdata[tid] = (gid < n) ? d_in[gid] : 0.0f;
    __syncthreads();

    // Interleaved addressing: stride grows — causes warp divergence
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    // Thread 0 writes this block's partial sum
    if (tid == 0) d_out[blockIdx.x] = sdata[0];
}

/*
 * Version B: Sequential (Converging) Tree Reduction
 * Stride shrinks each step (BLOCK_SIZE/2 -> ... -> 1).
 * Active threads satisfy: tid < stride
 * Active threads are always contiguous from index 0, so they fill
 * complete warps with no divergence — key improvement over Version A.
 * Each block writes its partial sum to d_out[blockIdx.x].
 */
__global__ void kernel_sequential(const float* __restrict__ d_in,
                                  float* __restrict__ d_out, int n)
{
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Load element; pad with 0 if out of bounds
    sdata[tid] = (gid < n) ? d_in[gid] : 0.0f;
    __syncthreads();

    // Converging addressing: stride halves — no warp divergence
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    // Thread 0 writes this block's partial sum
    if (tid == 0) d_out[blockIdx.x] = sdata[0];
}

/*
 * Version C: Warp-Shuffle Reduction
 * Two-phase approach avoiding shared memory for intra-warp communication:
 *
 * Phase 1 — Intra-warp: __shfl_down_sync reduces 32 lanes to 1 value
 *   using offsets 16, 8, 4, 2, 1 (5 steps = log2(32)).
 *   Thread i reads the register of thread i+offset directly — no
 *   shared memory or __syncthreads needed within a warp.
 *
 * Phase 2 — Inter-warp: lane 0 of each warp writes to a small shared
 *   array (8 slots for 8 warps). The first warp loads those 8 values
 *   and performs a second shuffle pass, producing the block sum in
 *   lane 0 of warp 0, which is written to d_out[blockIdx.x].
 */

// Helper: reduce val across all 32 lanes of a warp using shuffle
__device__ __forceinline__ float warp_reduce(float val) {
    unsigned mask = 0xffffffff; // all 32 lanes participate
    val += __shfl_down_sync(mask, val, 16);
    val += __shfl_down_sync(mask, val, 8);
    val += __shfl_down_sync(mask, val, 4);
    val += __shfl_down_sync(mask, val, 2);
    val += __shfl_down_sync(mask, val, 1);
    return val; // result valid only in lane 0
}

__global__ void kernel_warp(const float* __restrict__ d_in,
                            float* __restrict__ d_out, int n)
{
    // Shared memory only for the 8 per-warp partial sums (BLOCK_SIZE/32)
    __shared__ float warp_sums[BLOCK_SIZE / 32];

    int tid  = threadIdx.x;
    int gid  = blockIdx.x * blockDim.x + tid;
    int lane = tid % 32;   // lane index within warp (0-31)
    int wid  = tid / 32;   // warp index within block (0-7)

    // Each thread loads its element
    float val = (gid < n) ? d_in[gid] : 0.0f;

    // Phase 1: intra-warp reduction via shuffle
    val = warp_reduce(val);

    // Lane 0 of each warp saves its warp's result to shared memory
    if (lane == 0) warp_sums[wid] = val;
    __syncthreads(); // ensure all warp_sums written before phase 2

    // Phase 2: first warp reduces the 8 per-warp sums
    if (wid == 0) {
        val = (lane < (blockDim.x / 32)) ? warp_sums[lane] : 0.0f;
        val = warp_reduce(val);
        // Lane 0 of warp 0 holds the entire block's sum
        if (lane == 0) d_out[blockIdx.x] = val;
    }
}

/*
 * Generic second-pass kernel (used by all three versions).
 * Reduces an array of partial block sums down to a single value
 * using the same converging-stride approach as Version B.
 * Called iteratively until only one block remains.
 */
__global__ void kernel_final_reduce(float* d_partial, int n_partial)
{
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (gid < n_partial) ? d_partial[gid] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0) d_partial[blockIdx.x] = sdata[0];
}

// ── Validation ───────────────────────────────────────────────────────────────
void check_result(float cpu_res, float gpu_res) {
    float abs_diff    = fabsf(cpu_res - gpu_res);
    float rel_error   = abs_diff / fabsf(cpu_res);
    const float eps   = 1e-3f; // 0.1% relative tolerance

    if (rel_error < eps) {
        printf("  [PASS] Results match within %.1f%% relative error.\n"
               "         GPU: %f  CPU: %f  (rel err: %.6f%%)\n",
               eps * 100.0f, gpu_res, cpu_res, rel_error * 100.0f);
    } else {
        printf("  [FAIL] Results differ beyond tolerance.\n"
               "         GPU: %f  CPU: %f  (rel err: %.6f%%, limit: %.1f%%)\n",
               gpu_res, cpu_res, rel_error * 100.0f, eps * 100.0f);
    }
}

// ── Runner: reduce_interleaved ────────────────────────────────────────────────
float run_reduce_interleaved(const float* h_in, int n) {
    // Note: We are only timing the kernel execution, not the memory allocation/transfer
    // Required memory needs to be allocated/freed and transferred from/to the device

    float result = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate device memory for input and partial sums
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float *d_in, *d_partial;
    CHECK(cudaMalloc(&d_in,      (size_t)n      * sizeof(float)));
    CHECK(cudaMalloc(&d_partial, (size_t)blocks * sizeof(float)));
    CHECK(cudaMemcpy(d_in, h_in, (size_t)n * sizeof(float), cudaMemcpyHostToDevice));

    cudaEventRecord(start);
    
    // Pass 1: each block reduces its chunk into one partial sum
    kernel_interleaved<<<blocks, BLOCK_SIZE>>>(d_in, d_partial, n);

    // Pass 2+: iteratively reduce partial sums until one value remains
    int remaining = blocks;
    while (remaining > 1) {
        int next = (remaining + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kernel_final_reduce<<<next, BLOCK_SIZE>>>(d_partial, remaining);
        remaining = next;
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time = 0.0f;
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("  GPU Kernel Execution Time: %.3f ms\n", gpu_time);

    // Transfer final result back to host and free device memory
    CHECK(cudaMemcpy(&result, d_partial, sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_in);
    cudaFree(d_partial);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return result;
}

// ── Runner: reduce_sequential ─────────────────────────────────────────────────
float run_reduce_sequential(const float* h_in, int n) {
    // Note: We are only timing the kernel execution, not the memory allocation/transfer
    // Required memory needs to be allocated/freed and transferred from/to the device

    float result = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate device memory for input and partial sums
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float *d_in, *d_partial;
    CHECK(cudaMalloc(&d_in,      (size_t)n      * sizeof(float)));
    CHECK(cudaMalloc(&d_partial, (size_t)blocks * sizeof(float)));
    CHECK(cudaMemcpy(d_in, h_in, (size_t)n * sizeof(float), cudaMemcpyHostToDevice));

    cudaEventRecord(start);
    
    // Pass 1: each block reduces its chunk into one partial sum
    kernel_sequential<<<blocks, BLOCK_SIZE>>>(d_in, d_partial, n);

    // Pass 2+: iteratively reduce partial sums until one value remains
    int remaining = blocks;
    while (remaining > 1) {
        int next = (remaining + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kernel_final_reduce<<<next, BLOCK_SIZE>>>(d_partial, remaining);
        remaining = next;
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time = 0.0f;
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("  GPU Kernel Execution Time: %.3f ms\n", gpu_time);

    // Transfer final result back to host and free device memory
    CHECK(cudaMemcpy(&result, d_partial, sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_in);
    cudaFree(d_partial);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return result;
}

// ── Runner: reduce_warp ───────────────────────────────────────────────────────
float run_reduce_warp(const float* h_in, int n) {
    // Note: We are only timing the kernel execution, not the memory allocation/transfer
    // Required memory needs to be allocated/freed and transferred from/to the device

    float result = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate device memory: d_out holds one partial sum per block
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in,  (size_t)n      * sizeof(float)));
    CHECK(cudaMalloc(&d_out, (size_t)blocks * sizeof(float)));
    CHECK(cudaMemcpy(d_in, h_in, (size_t)n * sizeof(float), cudaMemcpyHostToDevice));

    cudaEventRecord(start);
    
    // Pass 1: each block produces one partial sum via warp shuffle
    kernel_warp<<<blocks, BLOCK_SIZE>>>(d_in, d_out, n);

    // Pass 2+: iteratively reduce partial sums until one value remains
    int remaining = blocks;
    while (remaining > 1) {
        int next = (remaining + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kernel_final_reduce<<<next, BLOCK_SIZE>>>(d_out, remaining);
        remaining = next;
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time = 0.0f;
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("  GPU Kernel Execution Time: %.3f ms\n", gpu_time);

    // Transfer final result back to host and free device memory
    CHECK(cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_in);
    cudaFree(d_out);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return result;
}

// ── Main Program ─────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <method> <size_mb>\n", argv[0]);
        fprintf(stderr, "Methods: 0=CPU, 1=Interleaved, 2=Sequential, 3=Warp\n");
        return 1;
    }

    int method  = atoi(argv[1]);
    int size_mb = atoi(argv[2]);

    if (method < 0 || method > 3) {
        fprintf(stderr, "Invalid method. Choose 0-3.\n"); return 1;
    }

    size_t bytes = (size_t)size_mb * 1024 * 1024;
    int n = bytes / sizeof(float);

    printf("=========================================================\n");
    printf("Array Size: %d MB (%d floats)\n", size_mb, n);
    printf("Method:     %d\n", method);
    printf("=========================================================\n");

    // Allocate and initialize host memory
    float* h_in = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) {
        h_in[i] = (float)(rand() % 100) / 100.0f;
    }

    // CPU reference (always computed for validation)
    float cpu_result = cpu_reduce(h_in, n);

    if (method == 0) {
        // ── CPU path ──────────────────────────────────────────────────────
        struct timespec wall0, wall1;
        clock_gettime(CLOCK_MONOTONIC, &wall0);
        float final_result = cpu_reduce(h_in, n);
        clock_gettime(CLOCK_MONOTONIC, &wall1);

        float cpu_ms = (wall1.tv_sec  - wall0.tv_sec)  * 1e3f
                     + (wall1.tv_nsec - wall0.tv_nsec) * 1e-6f;
        printf("  CPU Execution Time: %.3f ms\n", cpu_ms);
        printf("  Result: %f\n", final_result);

    } else {
        // ── GPU path ──────────────────────────────────────────────────────

        // Single dummy warmup
        float* d_dummy;
        CHECK(cudaMalloc(&d_dummy, sizeof(float)));
        warmup_dummy<<<1, 1>>>(d_dummy);
        CHECK(cudaDeviceSynchronize());
        cudaFree(d_dummy);

        float final_result = 0.0f;
        
        // Pass the host array (h_in) directly to the runners
        if      (method == 1) final_result = run_reduce_interleaved(h_in, n);
        else if (method == 2) final_result = run_reduce_sequential(h_in, n);
        else if (method == 3) final_result = run_reduce_warp(h_in, n);

        check_result(cpu_result, final_result);
    }

    printf("=========================================================\n");
    free(h_in);
    return 0;
}
