"""
Experiment 3: Filter Size Variation
Test all methods with different filter sizes (1x1, 3x3, 5x5, 9x9)
Keep thread count constant at number of physical cores (16)
For work queue, use chunk size = N (number of threads)
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

def run_timing_experiment(filter_type, method, numthreads, chunk_size, input_file, repeat=10):
    """Run timing experiment for specified configuration"""
    
    methods = {
        "sequential" : 1,
        "sharded_rows": 2,
        "sharded_columns_column_major" : 3,
        "sharded_columns_row_major" : 4,
        "work_queue" : 5,
    }
    
    filters = {"1x1" : 4, "3x3" : 1, "5x5" : 2, "9x9" : 3}
    
    main_args = f'./build/main -t 1 -i {input_file} -f {filters[filter_type]} -m {methods[method]} -n {numthreads} -c {chunk_size}'
    
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
        'time_data': time_data,
        'filter_size': int(filter_type.split('x')[0])
    }

def main():
    print("=== Experiment 3: Filter Size Variation ===")
    
    # Create test image
    create_image_command = "./build/pgm_creator 3000 7000 input.pgm"
    execute_command(create_image_command)
    image_file = "input.pgm"
    
    # Test configuration
    methods = ["sequential", "sharded_rows", "sharded_columns_column_major", 
               "sharded_columns_row_major", "work_queue"]
    filter_types = ["1x1", "3x3", "5x5", "9x9"]
    numthreads = 16  # Number of physical cores
    
    results = {}
    
    print(f"Running filter variation experiments with {numthreads} threads...")
    for method in methods:
        print(f"Testing method: {method}")
        for filter_type in filter_types:
            print(f"  Filter: {filter_type}")
            
            # For work queue, chunk size = numthreads; for others, use numthreads as well
            chunk_size = numthreads
            
            result = run_timing_experiment(filter_type, method, numthreads, chunk_size, image_file)
            results[(method, filter_type)] = result
    
    # Save results
    with open('experiment3_results.pickle', 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
    
    # Generate plots
    colors = {'sequential': 'red', 'sharded_rows': 'blue', 
              'sharded_columns_column_major': 'orange', 
              'sharded_columns_row_major': 'brown', 'work_queue': 'gray'}
    
    # Extract filter dimensions for x-axis
    filter_dims = [1, 3, 5, 9]
    
    # Plot 1: Execution Time vs Filter Size
    plt.figure(figsize=(14, 10))
    
    for method in methods:
        times = []
        errors = []
        
        for filter_type in filter_types:
            key = (method, filter_type)
            if key in results:
                times.append(results[key]['time'])
                errors.append(results[key]['time_std'])
            else:
                times.append(0)
                errors.append(0)
        
        plt.errorbar(filter_dims, times, yerr=errors, 
                    label=method.replace('_', ' ').title(), 
                    color=colors[method], marker='o', linewidth=2)
    
    plt.xlabel('Filter Dimension (NxN)')
    plt.ylabel('Execution Time (seconds)')
    plt.title(f'Performance vs Filter Size (3000x7000 image, {numthreads} threads)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(filter_dims, [f'{d}x{d}' for d in filter_dims])
    plt.tight_layout()
    plt.savefig('experiment3_filter_performance.png')
    plt.close()
    
    # Calculate theoretical work (proportional to filter area)
    filter_areas = [d*d for d in filter_dims]

    # Plot 2: Efficiency Analysis - Time per operation
    plt.figure(figsize=(14, 10))
    
    image_pixels = 3000 * 7000  # Total pixels in image
    
    for method in methods:
        efficiency_ratios = []
        
        for i, filter_type in enumerate(filter_types):
            key = (method, filter_type)
            if key in results:
                total_ops = image_pixels * filter_areas[i]  # Total operations
                time_per_op = results[key]['time'] / total_ops * 1e9  # nanoseconds per operation
                efficiency_ratios.append(time_per_op)
            else:
                efficiency_ratios.append(0)
        
        plt.plot(filter_dims, efficiency_ratios, 
                label=method.replace('_', ' ').title(), 
                color=colors[method], marker='o', linewidth=2)
    
    plt.xlabel('Filter Dimension (NxN)')
    plt.ylabel('Time per Operation (nanoseconds)')
    plt.title(f'Computational Efficiency vs Filter Size ({numthreads} threads)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(filter_dims, [f'{d}x{d}' for d in filter_dims])
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('experiment3_filter_efficiency.png')
    plt.close()
    
    
    print("\nExperiment 3 completed. Generated graphs:")
    print("  - experiment3_filter_performance.png")
    print("  - experiment3_filter_efficiency.png")
    print("  - Results saved to experiment3_results.pickle")

if __name__ == "__main__":
    main()