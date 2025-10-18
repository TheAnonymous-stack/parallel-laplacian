"""
Experiment 2: Work Queue Chunk Size Analysis
Test work queue method with various chunk sizes for different thread counts
"""

import subprocess
import os
import math
import pickle
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def execute_command(command):
    print(f">>>>> Executing command {command}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, shell=True,
        universal_newlines=True)
    return_code = process.wait()
    output = process.stdout.read()

    if return_code == 1:
        print(f"failed to execute command = {command}")
        print(output)
        exit()

    return output

def get_std(data_points, mean):
    """Get standard deviation of the sample data_points"""
    if len(data_points) <= 1:
        return 0
    sum_of_error = 0
    for point in data_points:
        sum_of_error += (point - mean) * (point - mean)
    variance = sum_of_error / (len(data_points) - 1)
    return math.sqrt(variance)

def run_timing_experiment(filter_type, numthreads, chunk_size, input_file, repeat=10):
    """Run timing experiment for work queue with specific chunk size"""
    
    filters = {"3x3" : 1, "5x5" : 2, "9x9" : 3, "1x1" : 4}
    method = 5  # work_queue
    
    main_args = f'./build/main -t 1 -i {input_file} -f {filters[filter_type]} -m {method} -n {numthreads} -c {chunk_size}'
    
    time_data = []
    total_time = 0
    
    for i in range(repeat):
        ret = execute_command(main_args)
        time_val = float(ret[5:])
        time_data.append(time_val)
        total_time += time_val
    
    avg_time = total_time / repeat
    std_time = get_std(time_data, avg_time)
    
    return {
        'time': avg_time,
        'time_std': std_time,
        'time_data': time_data
    }

def main():
    print("=== Experiment 2: Work Queue Chunk Size Analysis ===")
    
    # Create test image
    create_image_command = "./build/pgm_creator 2000 1500 input.pgm"
    execute_command(create_image_command)
    image_file = "input.pgm"
    
    # Test different thread counts
    thread_counts = [1, 2, 4, 8, 16]
    filter_type = "9x9"
    
    # Generate chunk sizes
    # Only use chunks less than half the image dimension
    max_chunk = min(10000,7000) // 2 
    chunk_sizes = []
    chunk = 1
    while chunk <= max_chunk:
        chunk_sizes.append(chunk)
        chunk *= 2
    
    print(f"Testing chunk sizes: {chunk_sizes}")
    
    results = {}
    
    print("Running chunk size experiments...")
    for nthread in thread_counts:
        print(f"Testing with {nthread} threads")
        for chunk_size in chunk_sizes:
            print(f"  Chunk size: {chunk_size}x{chunk_size}")
            
            result = run_timing_experiment(filter_type, nthread, chunk_size, image_file)
            results[(nthread, chunk_size)] = result
    
    # Save results
    with open('experiment2_results.pickle', 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
    
    # Generate plots
    plt.figure(figsize=(14, 10))
    
    # Define colors for different thread counts
    colors = plt.cm.viridis(np.linspace(0, 1, len(thread_counts)))
    
    # Plot 1: Time vs Chunk Size for each thread count
    for i, nthread in enumerate(thread_counts):
        times = []
        errors = []
        valid_chunks = []
        
        for chunk_size in chunk_sizes:
            key = (nthread, chunk_size)
            if key in results:
                times.append(results[key]['time'])
                errors.append(results[key]['time_std'])
                valid_chunks.append(chunk_size)
        
        if times:
            plt.errorbar(valid_chunks, times, yerr=errors, 
                        label=f'{nthread} threads', color=colors[i],
                        marker='o', linewidth=2, capsize=5, capthick=2)
    
    plt.xlabel('Chunk Size (NxN)', fontsize=16)
    plt.ylabel('Execution Time (seconds)', fontsize=16)
    plt.title('Work Queue Performance vs Chunk Size (10000x7000 image, 9x9 filter)', fontsize=18)
    plt.legend(fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.xticks(chunk_sizes, [f'{c}x{c}' for c in chunk_sizes], rotation=45)
    plt.tight_layout()
    plt.savefig('experiment2_chunk_size_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

    
    print("\nExperiment 2 completed. Generated graphs:")
    print("  - experiment2_chunk_size_performance.png")
    print("  - Results saved to experiment2_results.pickle")

if __name__ == "__main__":
    main()