# CUDA Parallel Reduction Optimization

Performance analysis and optimization of parallel reduction algorithms using NVIDIA CUDA. This project compares multiple GPU reduction strategies against a sequential CPU implementation and investigates their performance using NVIDIA Nsight Systems.

## Overview

Parallel reduction is a fundamental operation in GPU computing and scientific applications. The project implements and evaluates three CUDA-based reduction strategies with different approaches to thread cooperation and memory access.

The implementations are benchmarked across input sizes from **2 MB to 4096 MB** and profiled to investigate kernel execution time, memory transfers, GPU utilization, synchronization overhead, and warp efficiency.

## Implementations

### Version A — Interleaved Tree Reduction

A shared-memory reduction using interleaved thread addressing.

This implementation serves as the baseline GPU approach. Its conditional execution pattern introduces **warp divergence**, reducing effective SIMD utilization.

### Version B — Sequential Tree Reduction

A shared-memory reduction using a converging stride:

`BLOCK_SIZE/2 → BLOCK_SIZE/4 → ... → 1`

The `tid < stride` condition keeps active threads contiguous within warps, eliminating the divergence present in the interleaved implementation.

### Version C — Warp-Shuffle Reduction

The optimized implementation uses CUDA warp-shuffle intrinsics for intra-warp communication.

The reduction consists of:

1. Intra-warp reduction using `__shfl_down_sync`
2. Inter-warp reduction using a small shared-memory buffer
3. Final reduction by the first warp

This approach reduces shared-memory traffic and synchronization overhead by performing most intra-warp communication directly through registers.

## Performance

The warp-shuffle implementation achieved approximately **3.1× lower execution time than the baseline GPU implementation** at the largest tested input size.

## Profiling

The CUDA implementations were profiled using **NVIDIA Nsight Systems**.

Example profiling command:

```bash
nsys profile --stats=true -t cuda -o profile_X ./ex3_reduce <method> 4096
```

Profiling was used to investigate:

* CUDA kernel execution time
* Host-to-device and device-to-host transfers
* GPU utilization
* CUDA execution timelines
* Synchronization overhead
* Performance differences between reduction strategies

The profiler reported no GPU-utilization problems during kernel execution.

## Technologies

* C++
* NVIDIA CUDA
* CUDA Kernels
* CUDA Warp Shuffle Intrinsics
* Shared Memory
* Parallel Reduction
* GPU Performance Optimization
* NVIDIA Nsight Systems
* Linux
* Git

## Project details:

This project explores practical GPU performance considerations including:

* Warp divergence
* SIMD/SIMT execution
* Warp-level primitives
* Shared-memory communication
* Register-level communication
* Synchronization overhead
* Memory transfers
* Kernel benchmarking
* GPU profiling
* Parallel algorithm optimization
