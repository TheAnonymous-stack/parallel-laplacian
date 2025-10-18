#!/bin/bash
# Running this script with as ./generate_graphs_script.sh should produce all
# the graphs shown in your report.  You can simply call some other python
# script, or any other scripting language you want.

# You should assume this script will be called from the source directory and
# that the binaries are generated inside a directory called "build".

echo "=== Automated Experiment Suite ==="
echo "This script runs all 4 experiments and generates graphs for the report."
echo ""

echo "Starting experiments..."
echo ""

# Experiment 1: Thread Scaling with Cache Analysis
echo "=== Running Experiment 1: Thread Scaling with Cache Analysis ==="
python3 experiment1_thread_scaling.py
if [ $? -eq 0 ]; then
    echo "Experiment 1 completed successfully"
    echo "Generating experiment1_report.txt..."
    python3 read_pickle_experiment1.py
else
    echo "Experiment 1 failed"
fi
echo ""

# Experiment 2: Work Queue Chunk Size Analysis  
echo "=== Running Experiment 2: Work Queue Chunk Size Analysis ==="
python3 experiment2_chunk_size.py
if [ $? -eq 0 ]; then
    echo "Experiment 2 completed successfully"
    echo "Generating experiment2_report.txt..."
    python3 read_pickle_experiment2.py
else
    echo "Experiment 2 failed"
fi
echo ""

# Experiment 3: Filter Size Variation
echo "=== Running Experiment 3: Filter Size Variation ==="
python3 experiment3_filter_variation.py
if [ $? -eq 0 ]; then
    echo "Experiment 3 completed successfully"
    echo "Generating experiment3_report.txt..."
    python3 read_pickle_experiment3.py
else
    echo "Experiment 3 failed"
fi
echo ""

# Experiment 4: Image Variation Testing
echo "=== Running Experiment 4: Image Variation Testing ==="
python3 experiment4_image_variation.py
if [ $? -eq 0 ]; then
    echo "Experiment 4 completed successfully"
    echo "Generating experiment4_report.txt..."
    python3 read_pickle_experiment4.py
else
    echo "Experiment 4 failed"
fi
echo ""

echo "=== Experiment Suite Complete ==="
echo ""
echo "Generated graphs:"
echo "Experiment 1 (Thread Scaling):"
echo "  - experiment1_time_scaling.png     (Performance vs thread count)"
echo "  - experiment1_speedup.png          (Speedup analysis)"
echo "  - experiment1_cache_misses.png     (L1 cache miss analysis)"
echo ""
echo "Experiment 2 (Chunk Size):"
echo "  - experiment2_chunk_size_performance.png  (Performance vs chunk size)"
echo ""
echo "Experiment 3 (Filter Variation):"
echo "  - experiment3_filter_performance.png      (Performance vs filter size)"
echo "  - experiment3_filter_efficiency.png       (Time per operation)"
echo ""
echo "Experiment 4 (Image Variation):"
echo "  - experiment4_image_performance.png       (Performance by image type)"
echo "  - experiment4_image_speedup.png          (Speedup vs image size)"
echo "  - experiment4_image_throughput.png       (Processing throughput)"
echo "Data files:"
echo "  - experiment1_results.pickle"
echo "  - experiment2_results.pickle"
echo "  - experiment3_results.pickle"
echo "  - experiment4_results.pickle"
echo ""
echo "Summary reports:"
echo "  - experiment1_report.txt"
echo "  - experiment2_report.txt"
echo "  - experiment3_report.txt"
echo "  - experiment4_report.txt"
echo ""

# Clean up any temporary files
rm -f data.pickle work_pool.pickle

