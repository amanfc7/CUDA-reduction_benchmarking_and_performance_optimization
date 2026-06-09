#!/bin/bash

# Compile the code
echo "Compiling ex3_reduce..."
nvcc -O3 -arch=sm_75 -o ex3_reduce exercise3_reduce.cu

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Compilation successful. Starting benchmarks..."

# 1. Initialize the CSV file and add the header
CSV_FILE="benchmark_data.csv"
echo "Size_MB,Method,Time_ms" > $CSV_FILE

METHODS=(0 1 2 3)
SIZES=(2 16 128 1024 4096)

# 2. Run the benchmarks and parse the output
for size in "${SIZES[@]}"; do
    for method in "${METHODS[@]}"; do
        
        # Capture the output of the C++ program
        output=$(./ex3_reduce $method $size)
        
        # Print the output to the terminal so you can still monitor it
        echo "$output"
        
        # Extract the numeric execution time using grep and awk
        # This targets the line containing "Execution Time:" and grabs the second-to-last column (the number)
        time_ms=$(echo "$output" | grep "Execution Time:" | awk '{print $(NF-1)}')
        
        # Append the parsed data as a new row in the CSV
        echo "$size,$method,$time_ms" >> $CSV_FILE
        
    done
done

echo "Benchmarking complete. Data successfully extracted and saved to $CSV_FILE"