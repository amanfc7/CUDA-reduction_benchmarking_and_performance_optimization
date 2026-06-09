#!/bin/bash

mkdir -p outputs_exercise2

sizes=(1 4 16 64 256 512)
host_types=(0 1 2 3 4)
device_types=(0 1)

for h in "${host_types[@]}"; do
  for d in "${device_types[@]}"; do

    # Skip invalid combination: mapped host memory + managed device memory
    if [ "$h" -eq 4 ] && [ "$d" -eq 1 ]; then
      continue
    fi

    for s in "${sizes[@]}"; do
      echo "Running host=$h device=$d size=${s}MB"
      ./mem_bench "$h" "$d" "$s" | tee "outputs_exercise2/h${h}_d${d}_${s}MB.txt"
    done

  done
done