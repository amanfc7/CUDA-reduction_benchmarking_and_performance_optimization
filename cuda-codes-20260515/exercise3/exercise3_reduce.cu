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

// TODO: Define and implement your kernels






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

    // TODO: Allocate memory and transfer data to device

    cudaEventRecord(start);
    
    // TODO: Kernel call(s) come here
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time = 0.0f;
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("  GPU Kernel Execution Time: %.3f ms\n", gpu_time);

    // TODO: Transfer result back to host and free device memory

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

    // TODO: Allocate memory and transfer data to device

    cudaEventRecord(start);
    
    // TODO: Kernel call(s) come here
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time = 0.0f;
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("  GPU Kernel Execution Time: %.3f ms\n", gpu_time);

    // TODO: Transfer result back to host and free device memory

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

    // TODO: Allocate memory and transfer data to device

    cudaEventRecord(start);
    
    // TODO: Kernel call(s) come here
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time = 0.0f;
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("  GPU Kernel Execution Time: %.3f ms\n", gpu_time);

    // TODO: Transfer result back to host and free device memory

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