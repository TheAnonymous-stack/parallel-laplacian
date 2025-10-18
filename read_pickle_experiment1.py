import pickle

file = "experiment1_results.pickle"

with open(file, 'rb') as f:
    results = pickle.load(f)

with open("experiment1_report.txt", 'w') as f:
    sequential_reported = False
    str_to_write = ""
    for key in results:
        if key[0] == 'sequential' and sequential_reported:
            continue
        
        dict = results[key]
        
        str_to_write += "--------------------------------\n"
        str_to_write += f"Method: {key[0]}\n"
        str_to_write += f"Thread Count: {key[1]}\n"
        str_to_write += f"Average runtime: {round(results[key]['time'], 2)}\n"
        str_to_write += f"Standard Deviation: {results[key]['time_std']}\n"
        str_to_write += f"L1 miss rate: {round(int(results[key]['l1d_loadmisses']) / int(results[key]['l1d_load']) * 100,2)} %\n"
        str_to_write += f"L1 loads: {results[key]['l1d_load']}\n"
        str_to_write += "\n"

        if key[0] == 'sequential':
            sequential_reported = True
    f.write(str_to_write)