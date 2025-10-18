import pickle

file = "experiment2_results.pickle"

with open(file, 'rb') as f:
    results = pickle.load(f)

with open("experiment2_report.txt", 'w') as f:
    
    str_to_write = ""
    for key in results:
        
        dict = results[key]
        str_to_write += "--------------------------------\n"
        str_to_write += f"Chunk size: {key[1]}\n"
        str_to_write += f"Thread Count: {key[0]}\n"
        str_to_write += f"Average runtime: {round(results[key]['time'], 2)}\n"
        str_to_write += f"Standard Deviation: {results[key]['time_std']}\n"
        str_to_write += "\n"

        
    f.write(str_to_write)