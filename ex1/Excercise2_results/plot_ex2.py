import os
import re
import pandas as pd
import matplotlib.pyplot as plt

RAW_DIR = "raw_outputs"
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

host_labels = {
    0: "pageable",
    1: "pinned",
    2: "write-combine",
    3: "managed-host",
    4: "mapped-zero-copy"
}

device_labels = {
    0: "device",
    1: "managed-device"
}

def extract_median(text, label):
    pattern = rf"{re.escape(label)}\s+mean=\s*[\d.]+\s*ms\s+median=\s*([\d.]+)\s*ms"
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None

def extract_average_pipeline(text):
    pattern = r"Average Time per Full Iteration \(Timed\):\s*([\d.]+)\s*ms"
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None

rows = []

for filename in os.listdir(RAW_DIR):
    if not filename.endswith(".txt"):
        continue

    m = re.match(r"h(\d+)_d(\d+)_(\d+)MB\.txt", filename)
    if not m:
        continue

    h = int(m.group(1))
    d = int(m.group(2))
    size = int(m.group(3))

    with open(os.path.join(RAW_DIR, filename), "r") as f:
        text = f.read()

    host_alloc = extract_median(text, "host alloc")
    device_alloc = extract_median(text, "device alloc")
    kernel = extract_median(text, "kernel")
    device_free = extract_median(text, "device free")
    host_free = extract_median(text, "host free")
    avg_pipeline = extract_average_pipeline(text)

    h2d = 0.0 if "N/A (zero-copy: no explicit H2D transfer)" in text else extract_median(text, "H2D transfer")
    d2h = 0.0 if "N/A (zero-copy: no explicit D2H transfer)" in text else extract_median(text, "D2H transfer")

    rows.append({
        "host_type": h,
        "device_type": d,
        "size_mb": size,
        "combination": f"{host_labels[h]} + {device_labels[d]}",
        "h2d": h2d,
        "kernel": kernel,
        "d2h": d2h,
        "host_alloc_free": host_alloc + host_free if host_alloc is not None and host_free is not None else None,
        "device_alloc_free": device_alloc + device_free if device_alloc is not None and device_free is not None else None,
        "average_pipeline": avg_pipeline
    })

df = pd.DataFrame(rows)
df = df.sort_values(["host_type", "device_type", "size_mb"])
df.to_csv("exercise2_results.csv", index=False)

print(df)
print("\nNumber of rows:", len(df))

def plot_metric(metric, ylabel, title, filename):
    plt.figure(figsize=(10, 6))

    for combo, group in df.groupby("combination"):
        group = group.sort_values("size_mb")
        plt.plot(group["size_mb"], group[metric], marker="o", label=combo)

    plt.xscale("log", base=2)
    plt.xlabel("Message Size (MB)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=300)
    plt.close()

plot_metric("h2d", "Median H2D Transfer Time (ms)", "H2D Transfer Time vs Message Size", "plot1_h2d_transfer.png")
plot_metric("kernel", "Median Kernel Execution Time (ms)", "Kernel Execution Time vs Message Size", "plot2_kernel_time.png")
plot_metric("d2h", "Median D2H Transfer Time (ms)", "D2H Transfer Time vs Message Size", "plot3_d2h_transfer.png")
plot_metric("host_alloc_free", "Median Host Allocation + Free Time (ms)", "Host Memory Management Overhead vs Message Size", "plot4_host_alloc_free.png")
plot_metric("device_alloc_free", "Median Device Allocation + Free Time (ms)", "Device Memory Management Overhead vs Message Size", "plot5_device_alloc_free.png")
plot_metric("average_pipeline", "Average Full Pipeline Time (ms)", "Overall Pipeline Average Time vs Message Size", "plot6_average_pipeline.png")

print("Saved exercise2_results.csv")
print("Saved plots in plots/")