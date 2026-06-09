
#!/bin/bash

# Configuration array for the 3 target TRANSFORM_ITERS sizes
CONFIGS=(1 50 200)

for iters in "${CONFIGS[@]}"
do
    echo "=================================================="
    echo " Starting Profiling for TRANSFORM_ITERS = ${iters}"
    echo "=================================================="
    
    # Dynamically replace the TRANSFORM_ITERS line inside your source file
    sed -i "s/#define TRANSFORM_ITERS.*/#define TRANSFORM_ITERS  ${iters}/" exercise1.cu
    
    # Compile the updated CUDA binary targeting the Tesla T4 architecture
    nvcc -O3 -arch=sm_75 -o exercise1_out exercise1.cu
    
    # Initialize or wipe the output log file for this specific metric run
    echo "=== RAW BENCHMARK RUNS FOR ITERS = ${iters} ===" > results_iters_${iters}.txt
    
    # Sequentially execute the binary 5 times and redirect stdout to append to the log file
    for i in {1..5}
    do
       echo "  -> Processing Iteration Run $i/5..."
       echo -e "\n--- RUN $i ---" >> results_iters_${iters}.txt
       ./exercise1_out >> results_iters_${iters}.txt
    done
    
    echo -e "Success! Logs saved to: results_iters_${iters}.txt\n"
done

echo "All 15 benchmarking passes have finished executing successfully."

