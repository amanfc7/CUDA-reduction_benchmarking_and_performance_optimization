import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parse_and_average(filename):
    data = {
        "CPU reference": [],
        "no-stream": [],
        "1": [],
        "2": [],
        "4": [],
        "8": []
    }
    
    with open(filename, 'r') as f:
        content = f.read()
        
    # Extract CPU and GPU execution metrics using regex parsing
    cpu_times = [float(t) for t in re.findall(r"=== CPU reference ===\s+Total time:\s+([\d.]+)\s+ms", content)]
    data["CPU reference"] = cpu_times
    
    no_stream_times = [float(t) for t in re.findall(r"no-stream\s+([\d.]+)", content)]
    data["no-stream"] = no_stream_times
    
    # Targets the start of the line, grabs the stream count, and the execution time.
    
    stream_runs = re.findall(r"^\s*(\d+)\s+([\d.]+)", content, re.MULTILINE)
    for stream_count, time_val in stream_runs:
        if stream_count in data:
            data[stream_count].append(float(time_val))

    summary_rows = []
    mean_cpu = np.mean(data["CPU reference"]) if data["CPU reference"] else 0
    summary_rows.append({"Configuration": "CPU reference", "Mean time (ms)": round(mean_cpu, 2), "Speedup vs CPU": "—"})
    
    def add_row(label, key):
        if data[key]:
            mean_time = np.mean(data[key])
            speedup = round(mean_cpu / mean_time, 2) if mean_time > 0 else 0
            summary_rows.append({
                "Configuration": label, 
                "Mean time (ms)": round(mean_time, 2), 
                "Speedup vs CPU": f"{speedup}x"
            })

    add_row("Version A, no streams", "no-stream")
    add_row("Version B, 1 stream", "1")
    add_row("Version B, 2 streams", "2")
    add_row("Version B, 4 streams", "4")
    add_row("Version B, 8 streams", "8")
    
    return pd.DataFrame(summary_rows)

# Dictionary to hold stream performance trends specifically for plotting
plot_trends = {}

# Automatically iterate through the generated outputs
for iters in [1, 50, 200]:
    raw_file = f"results_iters_{iters}.txt"
    csv_file = f"summary_table_iters_{iters}.csv"
    
    try:
        df_summary = parse_and_average(raw_file)
        df_summary.to_csv(csv_file, index=False)
        
        print(f"\n===== FINAL SUMMARY TABLE FOR TRANSFORM_ITERS = {iters} =====")
        print(df_summary.to_string(index=False))
        print(f"File Saved: {csv_file}")
        
        # Pull out stream execution values safely to pass to matplotlib
        stream_keys = ["1 stream", "2 streams", "4 streams", "8 streams"]
        times = []
        for key in stream_keys:
            matched_row = df_summary[df_summary["Configuration"].str.contains(key)]
            if not matched_row.empty:
                times.append(matched_row.iloc[0]["Mean time (ms)"])
        
        if len(times) == 4:
            plot_trends[iters] = times
            
    except FileNotFoundError:
        print(f"Error: Missing logs for iteration configuration: {iters}")

# Plot the trends for Version B across different stream counts and iteration configurations:
if plot_trends:
    fig = plt.figure(figsize=(10, 6))
    
    stream_labels = ['1', '2', '4', '8']
    styles = {1: {'color': 'crimson', 'marker': 'o'}, 
              50: {'color': 'royalblue', 'marker': 's'}, 
              200: {'color': 'forestgreen', 'marker': '^'}}
    
    for iters, mean_times in plot_trends.items():
        plt.plot(stream_labels, mean_times, 
                 label=f'TRANSFORM_ITERS = {iters}', 
                 color=styles[iters]['color'], 
                 marker=styles[iters]['marker'], 
                 linewidth=2, markersize=8)
    
    plt.title("CUDA Streams Scaling Performance Analysis", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Active Hardware Stream Count (N_Streams)", fontsize=12)
    plt.ylabel("Mean Execution Time (ms)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    
    plot_filename = "cuda_streams_comparison.png"
    fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\n[Plot Success] Comparison line graph exported as: {os.path.abspath(plot_filename)}")
    
    plt.show()
    plt.close(fig)