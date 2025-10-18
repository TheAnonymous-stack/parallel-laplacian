"""
Experiment 4: Image Variation Testing
Test all methods with different image sizes and characteristics
Create various images to understand how different methods perform under different conditions
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

def create_test_images():
    """Create various test images with different characteristics"""
    images = []
    
    # Small square image
    cmd = "./build/pgm_creator 1000 1000 small_square.pgm"
    execute_command(cmd)
    images.append(("small_square.pgm", "Small Square (1000x1000)", 1000*1000))

    # Medium standard image
    cmd = "./build/pgm_creator 7000 5000 medium_std.pgm"
    execute_command(cmd)
    images.append(("medium_std.pgm", "Medium Standard (7000x5000)", 7000*5000))

    # Tall rectangular image
    cmd = "./build/pgm_creator 6000 10000 tall_rect.pgm"
    execute_command(cmd)
    images.append(("tall_rect.pgm", "Tall Rectangle (6000x10000)", 6000*10000))

    # Wide rectangular image
    cmd = "./build/pgm_creator 10000 6000 wide_rect.pgm"
    execute_command(cmd)
    images.append(("wide_rect.pgm", "Wide Rectangle (10000x6000)", 10000*6000))
    
    # Large square image
    cmd = "./build/pgm_creator 10000 10000 large_square.pgm"
    execute_command(cmd)
    images.append(("large_square.pgm", "Large Square (10000x10000)", 10000*10000))
    
    return images

def run_timing_experiment(image_file, filter_type, method, numthreads, chunk_size, repeat=10):
    """Run timing experiment for specified configuration"""
    
    methods = {
        "sequential" : 1,
        "sharded_rows": 2,
        "sharded_columns_column_major" : 3,
        "sharded_columns_row_major" : 4,
        "work_queue" : 5,
    }
    
    filters = {"3x3" : 1, "5x5" : 2, "9x9" : 3, "1x1" : 4}
    
    main_args = f'./build/main -t 1 -i {image_file} -f {filters[filter_type]} -m {methods[method]} -n {numthreads} -c {chunk_size}'
    
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
    print("=== Experiment 4: Image Variation Testing ===")
    
    # Create test images
    print("Creating test images...")
    images = create_test_images()
    
    # Test configuration
    methods = ["sequential", "sharded_rows", "sharded_columns_column_major", 
               "sharded_columns_row_major", "work_queue"]
    filter_type = "3x3" 
    numthreads = 16  # Number of physical cores
    
    results = {}
    
    print(f"Running image variation experiments with {numthreads} threads...")
    for image_file, image_desc, pixel_count in images:
        print(f"Testing image: {image_desc}")
        for method in methods:
            print(f"  Method: {method}")
            
            chunk_size = numthreads
            
            result = run_timing_experiment(image_file, filter_type, method, numthreads, chunk_size)
            result['pixel_count'] = pixel_count
            result['image_desc'] = image_desc
            results[(image_file, method)] = result
    
    # Save results
    with open('experiment4_results.pickle', 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
    
    # Generate plots
    colors = {'sequential': 'red', 'sharded_rows': 'blue', 
              'sharded_columns_column_major': 'orange', 
              'sharded_columns_row_major': 'brown', 'work_queue': 'gray'}
    
    # Plot 1: Execution Time by Image Type
    fig, ax = plt.subplots(figsize=(16, 10))
    
    image_names = [image[1] for image in images]
    x_pos = np.arange(len(image_names))
    width = 0.15
    
    for i, method in enumerate(methods):
        times = []
        errors = []
        
        for image_file, image_desc, pixel_count in images:
            key = (image_file, method)
            if key in results:
                times.append(results[key]['time'])
                errors.append(results[key]['time_std'])
            else:
                times.append(0)
                errors.append(0)
        
        ax.bar(x_pos + i*width, times, width, yerr=errors,
               label=method.replace('_', ' ').title(), color=colors[method])
    
    ax.set_xlabel('Image Type', fontsize=16)
    ax.set_ylabel('Execution Time (seconds)', fontsize=16)
    ax.set_title(f'Performance vs Image Type (3x3 filter, {numthreads} threads)', fontsize=18)
    ax.set_xticks(x_pos + width * 2)
    ax.set_xticklabels(image_names, rotation=45, ha='right')
    ax.legend(fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('experiment4_image_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Speedup vs Image Size
    plt.figure(figsize=(14, 10))
    
    pixel_counts = [image[2] for image in images]
    
    for method in methods:
        if method == "sequential":
            continue
            
        speedups = []
        
        for image_file, image_desc, pixel_count in images:
            method_key = (image_file, method)
            seq_key = (image_file, "sequential")
            
            if method_key in results and seq_key in results:
                speedup = results[seq_key]['time'] / results[method_key]['time']
                speedups.append(speedup)
            else:
                speedups.append(1)
        
        plt.plot(pixel_counts, speedups, 
                label=method.replace('_', ' ').title(), 
                color=colors[method], marker='o', linewidth=2)
    
    plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Sequential Baseline')
    plt.xlabel('Image Size (pixels)', fontsize=16)
    plt.ylabel('Speedup vs Sequential', fontsize=16)
    plt.title(f'Speedup vs Image Size ({numthreads} threads)', fontsize=18)
    plt.legend(fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('experiment4_image_speedup.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Efficiency (pixels per second) vs Image Size
    plt.figure(figsize=(14, 10))
    
    for method in methods:
        throughputs = []
        
        for image_file, image_desc, pixel_count in images:
            key = (image_file, method)
            if key in results:
                throughput = pixel_count / results[key]['time']  # pixels per second
                throughputs.append(throughput)
            else:
                throughputs.append(0)
        
        plt.plot(pixel_counts, throughputs, 
                label=method.replace('_', ' ').title(), 
                color=colors[method], marker='o', linewidth=2)
    
    plt.xlabel('Image Size (pixels)', fontsize=16)
    plt.ylabel('Throughput (pixels/second)', fontsize=16)
    plt.title(f'Processing Throughput vs Image Size ({numthreads} threads)', fontsize=18)
    plt.legend(fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('experiment4_image_throughput.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nExperiment 4 completed. Generated graphs:")
    print("  - experiment4_image_performance.png")
    print("  - experiment4_image_speedup.png")
    print("  - experiment4_image_throughput.png")
    print("  - Results saved to experiment4_results.pickle")

if __name__ == "__main__":
    main()