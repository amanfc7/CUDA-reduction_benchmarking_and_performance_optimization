/*
 * Exercise: Parallelization with CUDA and usage of CUDA Streams
 * ===================================================================
 * GPU target: NVIDIA T4
 * Compile:    nvcc -O3 -arch=sm_75 -o exercise1_out exercise1.cu
 * Run:        ./exercise1_out
 * Profile:    nsys profile --stats=true -t cuda -o ex1_full ./exercise1_out
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

// ============================================================================
// EXPERIMENT PARAMETERS 
// ============================================================================
#define N_TOTAL          (1 << 28)    // 256 M elements (~1 GB at float32)
#define N_CHUNKS         16           // Number of chunks for the stream pipeline
#define CHUNK_SIZE       (N_TOTAL / N_CHUNKS) // Elements per chunk
#define TRANSFORM_ITERS  200

// ============================================================================
// DO NOT MODIFY THE FOLLOWING MACROS
// ============================================================================
#define BLOCK_SIZE       256
#define MAX_STREAMS      8           // Maximum stream pool size

typedef unsigned int uint;

#define CHECK(call) \
    do { \
        cudaError_t e = (call); \
        if (e != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d — %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(1); \
        } \
    } while(0)

// ============================================================================
// CPU baseline code (DO NOT MODIFY)
// ============================================================================
static inline uint transform(float x) {
    uint v;
    // Safely extract float bits into unsigned int to avoid Undefined Behavior
    memcpy(&v, &x, sizeof(uint)); 
    
    for (int i = 0; i < TRANSFORM_ITERS; i++) {
        v = (v * 1103515245U) + 12345U;
        v ^= (v >> 16);
    }
    return v % 100U;
}

// ============================================================================
// Device Kernels (Define your needed CUDA Kernels here!)
// ============================================================================

__global__ void warmup_kernel() { 
    // a warmup kernel to have more accurate time measurements
}

// [YOUR KERNELS GO HERE]
__global__ void transform_kernel(const float* d_in, uint* d_out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = d_in[idx];
        uint v;
        
        // Use a direct device register cast instead of memcpy
        v = *(reinterpret_cast<const uint*>(&x));
        
        for (int i = 0; i < TRANSFORM_ITERS; i++) {
            v = (v * 1103515245U) + 12345U;
            v ^= (v >> 16);
        }
        d_out[idx] = v % 100U;
    }
}


// ============================================================================
// The following functions to be completed (Do not modify the signatures)
// ============================================================================

// ----------------------------------------------------------------------------
// TODO: VERSION A (Synchronous, No Streams)
// Process the entire N_TOTAL array in a single batch.
// Note: You must allocate the required device memory and free it at the end.
//
// Parameters:
//   h_in  : Host memory containing the input floats (Size: N_TOTAL)
//   h_out : Host memory to store the final uints (Size: N_TOTAL)
// ----------------------------------------------------------------------------
void run_gpu_nostream(const float* h_in, uint* h_out) 
{
    // [YOUR CODE GOES HERE]
    float* d_in;
    uint* d_out;
    
    size_t in_bytes  = (size_t)N_TOTAL * sizeof(float);
    size_t out_bytes = (size_t)N_TOTAL * sizeof(uint);
    
    // Allocate device buffers
    CHECK(cudaMalloc(&d_in, in_bytes));
    CHECK(cudaMalloc(&d_out, out_bytes));
    
    // Synchronously copy input to device
    CHECK(cudaMemcpy(d_in, h_in, in_bytes, cudaMemcpyHostToDevice));
    
    // Launch execution grid
    int grid_size = (N_TOTAL + BLOCK_SIZE - 1) / BLOCK_SIZE;
    transform_kernel<<<grid_size, BLOCK_SIZE>>>(d_in, d_out, N_TOTAL);
    CHECK(cudaGetLastError());
    
    // Synchronously copy output back to host
    CHECK(cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost));
    
    // Compute the sequential inclusive prefix sum
    for (int i = 1; i < N_TOTAL; i++) {
        h_out[i] += h_out[i - 1];
    }
    
    // Clean up memory
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
}


// ----------------------------------------------------------------------------
// TODO: VERSION B (Asynchronous, Stream-Based)
// Process the N_TOTAL array in N_CHUNKS chunks. 
// Overlap Memory and Compute using Async memory copies.
//
// Parameters (Pre-allocated in main for you):
//    n_streams    : The number of active streams for this benchmark run
//    streams      : Array containing the initialized cudaStream_t objects
//    h_in  : Host memory containing the full input (Size: N_TOTAL)
//    h_out : Host memory to store the full output (Size: N_TOTAL)
// ----------------------------------------------------------------------------
void run_gpu_streams(int n_streams, cudaStream_t* streams,
                     const float* h_in, uint* h_out) 
{
    // [YOUR CODE GOES HERE]
    float* d_in;
    uint* d_out;
    
    size_t in_bytes  = (size_t)N_TOTAL * sizeof(float);
    size_t out_bytes = (size_t)N_TOTAL * sizeof(uint);
    
    size_t chunk_in_bytes  = (size_t)CHUNK_SIZE * sizeof(float);
    size_t chunk_out_bytes = (size_t)CHUNK_SIZE * sizeof(uint);
    
    // Allocate full buffers on device
    CHECK(cudaMalloc(&d_in, in_bytes));
    CHECK(cudaMalloc(&d_out, out_bytes));
    
    int grid_size = (CHUNK_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Pipeline loop over all chunks using streams
    for (int i = 0; i < N_CHUNKS; i++) {
        int s = i % n_streams; // Target stream index selection
        size_t offset = (size_t)i * CHUNK_SIZE;
        
        // Asynchronous copy of chunk from Host to Device
        CHECK(cudaMemcpyAsync(d_in + offset, h_in + offset, chunk_in_bytes, 
                              cudaMemcpyHostToDevice, streams[s]));
                              
        // Run kernel transformation on this stream chunk
        transform_kernel<<<grid_size, BLOCK_SIZE, 0, streams[s]>>>(d_in + offset, d_out + offset, CHUNK_SIZE);
        
        // Asynchronous copy of transformed chunk from Device back to Host
        CHECK(cudaMemcpyAsync(h_out + offset, d_out + offset, chunk_out_bytes, 
                              cudaMemcpyDeviceToHost, streams[s]));
    }
    
    // Wait for all streams to finish processing completely
    CHECK(cudaDeviceSynchronize());
    
    // Compute the sequential inclusive prefix sum over the entire populated host array
    for (int i = 1; i < N_TOTAL; i++) {
        h_out[i] += h_out[i - 1];
    }
    
    // Clean up memory
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
}


// ============================================================================
// Host code (Do not modify these functions)
// ============================================================================

// The main computation function containing transformation and prefix sum
void cpu_pipeline(const float* in, uint* out, int n) {
    for (int i = 0; i < n; i++) out[i] = transform(in[i]);
    for (int i = 1; i < n; i++) out[i] += out[i-1];
}

int results_match(const uint* a, const uint* b, int n) {
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            printf("  First mismatch at index %d: expected %u, got %u\n", i, a[i], b[i]);
            return 0;
        }
    }
    return 1;
}

void fill_random(float* h, int n) {
    for (int i = 0; i < n; i++) h[i] = ((float)rand() / RAND_MAX) * 200.0f - 100.0f;
}

int main(void) {
    // ========================================================================
    // 1. HARDWARE WARMUP & ALLOCATION
    // ========================================================================
    cudaFree(0); 
    warmup_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    printf("N_total:         %d (%zu MB float input)\n", N_TOTAL, (size_t)N_TOTAL * sizeof(float) / (1 << 20));
    printf("N_chunks:        %d\n", N_CHUNKS);
    printf("Transform Iters: %d\n\n", TRANSFORM_ITERS);

    size_t in_bytes  = (size_t)N_TOTAL * sizeof(float);
    size_t out_bytes = (size_t)N_TOTAL * sizeof(uint);

    float *h_in;  CHECK(cudaMallocHost(&h_in,  in_bytes));
    uint  *h_out; CHECK(cudaMallocHost(&h_out, out_bytes));
    
    // Fill memory with random data
    fill_random(h_in, N_TOTAL);

    // Create a standard pageable memory copy specifically for Version A
    float *h_in_p = (float*)malloc(in_bytes);
    memcpy(h_in_p, h_in, in_bytes);

    // ========================================================================
    // 2. CPU REFERENCE BENCHMARK
    // ========================================================================
    uint *h_out_ref = (uint*)malloc(out_bytes);
    printf("=== CPU reference ===\n");
    
    struct timespec cpu_t0, cpu_t1;
    clock_gettime(CLOCK_MONOTONIC, &cpu_t0);
    cpu_pipeline(h_in, h_out_ref, N_TOTAL);
    clock_gettime(CLOCK_MONOTONIC, &cpu_t1);
    
    double ms_cpu = (cpu_t1.tv_sec - cpu_t0.tv_sec) * 1e3 + (cpu_t1.tv_nsec - cpu_t0.tv_nsec) * 1e-6;
    printf("  Total time:  %.2f ms\n\n", ms_cpu);

    // ========================================================================
    // 3. VERSION A: NO Streams
    // ========================================================================
    uint *h_out_A = (uint*)malloc(out_bytes);

    printf("=== Version A: GPU, no streams ===\n");
    printf("%-14s  %-12s  %-10s  %-8s\n", "Version", "Time(ms)", "Speedup", "Correct");
    printf("%-14s  %-12s  %-10s  %-8s\n", "-------", "--------", "-------", "-------");

    cudaEvent_t ta0, ta1;
    cudaEventCreate(&ta0); cudaEventCreate(&ta1);
    
    memset(h_out_A, 0, out_bytes);
    
    cudaEventRecord(ta0);
    
    run_gpu_nostream(h_in_p, h_out_A);
    
    cudaEventRecord(ta1);
    cudaDeviceSynchronize(); // Explicit sync added for safety
    cudaEventSynchronize(ta1);
    
    float ms_A = 0; cudaEventElapsedTime(&ms_A, ta0, ta1);
    printf("%-14s  %-12.2f  %-10.2fx  %-8s\n", 
           "no-stream", ms_A, ms_cpu / ms_A, results_match(h_out_ref, h_out_A, N_TOTAL) ? "YES" : "NO");

    free(h_out_A);

    // ========================================================================
    // 4. VERSION B: CUDA Stream
    // ========================================================================
    cudaStream_t streams[MAX_STREAMS];
    for (int s = 0; s < MAX_STREAMS; s++) cudaStreamCreate(&streams[s]);

    printf("\n=== Version B: GPU, stream-based ===\n");
    printf("%-14s  %-12s  %-10s  %-8s\n", "N_streams", "Time(ms)", "Speedup", "Correct");
    printf("%-14s  %-12s  %-10s  %-8s\n", "---------", "--------", "-------", "-------");
    
    int stream_counts[] = {1, 2, 4, 8};
    
    for (int si = 0; si < 4; si++) {
        int ns = stream_counts[si];
        memset(h_out, 0, out_bytes); 
        
        cudaEventRecord(ta0);
        
        run_gpu_streams(ns, streams, h_in, h_out);
        
        cudaEventRecord(ta1);
        cudaEventSynchronize(ta1);
        
        float ms_B = 0; cudaEventElapsedTime(&ms_B, ta0, ta1);
        printf("%-14d  %-12.2f  %-10.2fx  %-8s\n", 
               ns, ms_B, ms_cpu / ms_B, results_match(h_out_ref, h_out, N_TOTAL) ? "YES" : "NO");
    }

    // ========================================================================
    // 5. GLOBAL CLEANUP
    // ========================================================================
    for (int s = 0; s < MAX_STREAMS; s++) cudaStreamDestroy(streams[s]);
    cudaFreeHost(h_in); 
    cudaFreeHost(h_out);
    free(h_in_p);
    free(h_out_ref);
    cudaEventDestroy(ta0); 
    cudaEventDestroy(ta1);

    return 0;
}