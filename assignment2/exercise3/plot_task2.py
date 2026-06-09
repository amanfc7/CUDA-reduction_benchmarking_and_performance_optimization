import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the dynamically generated CSV data
df = pd.read_csv('benchmark_data.csv')

# 2. Pivot the data so sizes are rows and methods are columns
pivot_df = df.pivot(index='Size_MB', columns='Method', values='Time_ms')

# 3. Create the plot
plt.figure(figsize=(10, 6))

plt.plot(pivot_df.index, pivot_df[0], marker='o', linewidth=2, label='CPU Reference')
plt.plot(pivot_df.index, pivot_df[1], marker='s', linewidth=2, label='Version A: Interleaved Tree')
plt.plot(pivot_df.index, pivot_df[2], marker='^', linewidth=2, label='Version B: Sequential Tree')
plt.plot(pivot_df.index, pivot_df[3], marker='d', linewidth=2, label='Version C: Warp-Shuffle')

# 4. Format the plot according to HPC standards
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel('Array Size (MB) [Log Scale]')
plt.ylabel('Execution Time (ms) [Log Scale]')
plt.title('Exercise 3: Parallel Reduction Performance Comparison')

# Ensure X-axis ticks exactly match the array sizes
plt.xticks(pivot_df.index, labels=[str(int(x)) for x in pivot_df.index])

plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()

# 5. Save the output
plt.savefig('task2_plot.png', dpi=300, bbox_inches='tight')
print("Graph dynamically generated and saved as 'task2_plot.png'")