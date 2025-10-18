import pickle

file = "experiment3_results.pickle"

with open(file, 'rb') as f:
    results = pickle.load(f)

with open("experiment3_report.txt", 'w') as f:
    
    str_to_write = ""
    for key in results:
        
        dict = results[key]
        str_to_write += "--------------------------------\n"
        str_to_write += f"Method: {key[0]}\n"
        str_to_write += f"Filter type: {key[1]}\n"
        str_to_write += f"Average runtime: {round(results[key]['time'], 2)}\n"
        str_to_write += f"Standard Deviation: {results[key]['time_std']}\n"
        str_to_write += "\n"

        
    f.write(str_to_write)