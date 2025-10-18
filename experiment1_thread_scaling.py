"""
Experiment 1: Thread Scaling with Cache Analysis
Run sequential and parallel methods with varying thread counts (1, 2, 4, 8, 16)
Analyze performance scaling and L1 cache miss behavior
"""

import subprocess
import os
import re
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

def parse_perf(x):
    dict = {}
    dict['dump'] = x
    x = x.replace(',', '')
    
    # Patterns for CPU output inclduing both cpu_atom and cpu_core
    patterns = {
        'instructions': [
            r'(\d+)\s+cpu_core/instructions:u/',
            r'(\d+)\s+cpu_atom/instructions:u/'
        ],
        'l1d_load': [
            r'(\d+)\s+cpu_core/L1-dcache-loads/',
            r'(\d+)\s+cpu_atom/L1-dcache-loads/'
        ],
        'l1d_loadmisses': [
            r'(\d+)\s+cpu_core/L1-dcache-load-misses/',
            r'(\d+)\s+cpu_atom/L1-dcache-load-misses/'
        ],
        'll_load': [
            r'(\d+)\s+cpu_core/LLC-loads/',
            r'(\d+)\s+cpu_atom/LLC-loads/'
        ],
        'll_loadmisses': [
            r'(\d+)\s+cpu_core/LLC-load-misses/',
            r'(\d+)\s+cpu_atom/LLC-load-misses/'
        ]
    }
    
    for key, pattern_list in patterns.items():
        total = 0
        found_any = False
        
        for pattern in pattern_list:
            matches = re.findall(pattern, x)
            for match in matches:
                if match.strip() and match != '<not supported>':
                    total += int(match)
                    found_any = True
        
        if found_any:
            dict[key] = str(total)
    
    return dict

def get_std(data_points, mean):
    """Get standard deviation of the sample data_points"""
    if len(data_points) <= 1:
        return 0
    sum_of_error = 0
    for point in data_points:
        sum_of_error += (point - mean) * (point - mean)
    variance = sum_of_error / (len(data_points) - 1)
    return math.sqrt(variance)

def run_perf_experiment(filter_type, method, numthreads, chunk_size, input_file, repeat=10):
    """Run performance experiment with perf counters and timing"""
    
    methods = {
        "sequential" : 1,
        "sharded_rows": 2,
        "sharded_columns_column_major" : 3,
        "sharded_columns_row_major" : 4,
        "work_queue" : 5,
    }
    
    filters = {"3x3" : 1, "5x5" : 2, "9x9" : 3, "1x1" : 4}
    
    main_args = f'./build/main -t 0 -i {input_file} -f {filters[filter_type]} -m {methods[method]} -n {numthreads} -c {chunk_size}'

    # Cold run
    command = f'perf stat {main_args}'
    ret = execute_command(command)

    # Actual run: capture perf counters
    counters = [
         'instructions:u',
         'L1-dcache-loads:u',
         'L1-dcache-load-misses:u',
         'LLC-loads:u',
         'LLC-load-misses:u',
         ]
    groups_of_four_counters = [counters[i:i+4] for i in range(0, len(counters), 4)]
    partial_results = {}
    
    for counter_group in groups_of_four_counters:
        command = f'perf stat -r {repeat} -e {",".join(counter_group)} {main_args}'
        ret = execute_command(command)
        parsed = parse_perf(ret)
        partial_results = {**parsed, **partial_results}
        if 'dump' in partial_results and 'dump' in parsed:
            if partial_results['dump'] != parsed['dump']:
                partial_results['dump'] += '\n' + parsed['dump']
        
        

    # Get timing data
    main_args_time = f'./build/main -t 1 -i {input_file} -f {filters[filter_type]} -m {methods[method]} -n {numthreads} -c {chunk_size}'
    
    time_data = []
    total_time = 0
    for i in range(repeat):
        ret = execute_command(main_args_time)
        time_val = float(ret[5:])
        time_data.append(time_val)
        total_time += time_val
    
    avg_time = total_time / repeat
    std_time = get_std(time_data, avg_time)
    
    partial_results['time'] = avg_time
    partial_results['time_std'] = std_time
    partial_results['time_data'] = time_data
    
    return partial_results

def plot_with_error_bars(x_data, y_data, y_errors, labels, colors, filename, title, xlabel, ylabel):
    """Create plot with error bars"""
    plt.figure(figsize=(12, 8))
    
    for i, (y_vals, y_errs, label, color) in enumerate(zip(y_data, y_errors, labels, colors)):
        plt.errorbar(x_data, y_vals, yerr=y_errs, label=label, color=color, 
                    marker='o')
    
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.title(title, fontsize=18)
    plt.legend(fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("=== Experiment 1: Thread Scaling with Cache Analysis ===")
    
    # Create test image
    create_image_command = "./build/pgm_creator 4000 2500 input.pgm"
    execute_command(create_image_command)
    image_file = "input.pgm"
    
    methods = ["sequential","sharded_rows", "sharded_columns_column_major", 
               "sharded_columns_row_major", "work_queue"]
    threads = [1, 2, 4, 8, 16]
    filter_type = "5x5"
    
    results = {}
    
    print("Running thread scaling experiments...")
    for method in methods:
        print(f"Testing method: {method}")
        for nthread in threads:
            chunk_size = nthread if method == "work_queue" else nthread
            print(f"  Thread count: {nthread}")
            
            result = run_perf_experiment(filter_type, method, nthread, chunk_size, image_file)
            results[(method, nthread)] = result
    
    # Save results
    with open('experiment1_results.pickle', 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
    
    # Generate plots
    colors = {'sequential': 'red', 'sharded_rows': 'blue', 
              'sharded_columns_column_major': 'orange', 
              'sharded_columns_row_major': 'brown', 'work_queue': 'gray'}
    
    # Plot 1: Execution Time with Error Bars
    time_data = []
    time_errors = []
    labels = []
    plot_colors = []
    
    for method in methods:
        times = []
        errors = []
        for nthread in threads:
            key = (method, nthread)
            if key in results:
                times.append(results[key]['time'])
                errors.append(results[key]['time_std'])
            else:
                times.append(0)
                errors.append(0)
        time_data.append(times)
        time_errors.append(errors)
        labels.append(method.replace('_', ' ').title())
        plot_colors.append(colors[method])
    
    plot_with_error_bars(threads, time_data, time_errors, labels, plot_colors,
                        'experiment1_time_scaling.png',
                        'Thread Scaling Performance (4000x2500 image, 5x5 filter)',
                        'Number of Threads', 'Execution Time (seconds)')
    
    # Plot 2: Speedup
    plt.figure(figsize=(12, 8))
    for method in methods:
        if method == "sequential":
            continue
        speedups = []
        for nthread in threads:
            key = (method, nthread)
            seq_key = ("sequential", 1)
            if key in results and seq_key in results:
                speedup = results[seq_key]['time'] / results[key]['time']
                speedups.append(speedup)
            else:
                speedups.append(1)
        
        plt.plot(threads, speedups, marker='o', linewidth=2, 
                label=method.replace('_', ' ').title(), color=colors[method])
    
    # Add ideal speedup line
    plt.plot(threads, threads, 'k--', alpha=0.5, label='Ideal Speedup')
    
    plt.xlabel('Number of Threads', fontsize=16)
    plt.ylabel('Speedup', fontsize=16)
    plt.title('Speedup vs Number of Threads', fontsize=18)
    plt.legend(fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('experiment1_speedup.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: L1 Cache Miss Rate
    plt.figure(figsize=(12, 8))
    for method in methods:
        miss_rates = []
        for nthread in threads:
            key = (method, nthread)
            if key in results and 'l1d_loadmisses' in results[key] and 'l1d_load' in results[key]:
                misses = float(results[key]['l1d_loadmisses'])
                loads = float(results[key]['l1d_load'])
                miss_rate = (misses / loads) * 100 if loads > 0 else 0
                miss_rates.append(miss_rate)
            else:
                miss_rates.append(0)
        
        if any(rate > 0 for rate in miss_rates):
            plt.plot(threads, miss_rates, marker='o', linewidth=2, 
                    label=method.replace('_', ' ').title(), color=colors[method])
    
    plt.xlabel('Number of Threads', fontsize=16)
    plt.ylabel('L1 Cache Miss Rate (%)', fontsize=16)
    plt.title('L1 Cache Miss Rate vs Number of Threads', fontsize=18)
    plt.legend(fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('experiment1_cache_misses.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Experiment 1 completed. Generated graphs:")
    print("  - experiment1_time_scaling.png")
    print("  - experiment1_speedup.png") 
    print("  - experiment1_cache_misses.png")
    print("  - Results saved to experiment1_results.pickle")

if __name__ == "__main__":
    main()