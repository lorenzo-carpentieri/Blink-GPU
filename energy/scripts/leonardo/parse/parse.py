import argparse
from pathlib import Path
import pandas as pd
import re
from itertools import zip_longest
from statistics import median

msg_col="message_size_bytes"
run_col="run_id"
device_energy_col="device_energy_mj_1coll"
host_energy_col="host_energy_mj_1coll" 
goodput_col="goodput_Gb_per_s"
time_col="time_ms_1coll"

header_cols = [
    "Collective",
    "Library",
    "Data type",
    "Message size (B)",
    "Num. nodes", # Total number of nodes
    "Num. GPUs", # Total number of GPUs
    "GPUs per node",
    "Protocol",
    "Algorithm",
    "Channels",
    "Threads", 
    "Time (ms)", # max time between all ranks
    "Device Energy (mJ)", # sum of all devices
    "Host Energy (mJ)",
    "Goodput (Gb/s)"
]

def parse_power_trace(s):
    matches = re.findall(r'\(([^;]+);([^)]+)\)', s)
    return [(t, int(v)) for t, v in matches]


def parse_filename(s: str):
    tokens = s.split("_")

    result = {
        "coll": tokens[0],
        "alg": None,
        "prot": None,
        "ch": None,
        "th": None,
    }

    for token in tokens:
        if token.startswith("alg"):
            result["alg"] = token[3:]
        elif token.startswith("prot"):
            result["prot"] = token[4:]
        elif token.startswith("ch"):
            result["ch"] = int(token[2:])
        elif token.startswith("th"):
            result["th"] = int(token[2:])

    return result



def median_tuples(data):
    result = []

    for group in zip_longest(*data, fillvalue=None):
        xs = [t[0] for t in group if t is not None]
        ys = [t[1] for t in group if t is not None]

        if xs and ys:  # avoid empty case
            result.append((median(xs), median(ys)))

    return result

def process_csv(csv_path: Path, out_folder: Path):
    file_name_info = parse_filename(csv_path.name)
    # Extract info about coll, algorithm, protocol, channels and threads
    coll_name=file_name_info["coll"]
    alg=file_name_info["alg"]
    prot=file_name_info["prot"]
    ch=file_name_info["ch"]
    th=file_name_info["th"]
    
    # coll_name, scaleup, scaleout = parse_filename(csv_path.name)

    df = pd.read_csv(csv_path)
    num_nodes = df["total_nodes"].values[0]
    num_gpus = df["num_ranks"].values[0]
    
    
    # Iterate by message size
    msg_sizes = df[msg_col].unique()
  
    rows = []
    for data_type in df["data_type"].unique():
        df_data_type = df[df["data_type"] == data_type]
        for msg_size in msg_sizes:
            df_msg = df_data_type[df_data_type[msg_col] == msg_size]

            runs = df_msg[run_col].unique() # iterate by run_id
            times_ms=[]
            dev_energies_mj=[]
            goodputs_gbps=[]
            power_traces = []
            for run in runs:
                df_run = df_msg[df_msg[run_col] == run]
                power_trace = df_run['power_trace'].apply(parse_power_trace).values[run]
                power_traces.append(power_trace)
            
                # Extract relevant metrics for this run
                time_ms = max(df_run[time_col].values)
                dev_energy_mj = sum(df_run[device_energy_col].values)
                goodput_gbps = min(df_run[goodput_col]) 
                
                times_ms.append(time_ms)
                dev_energies_mj.append(dev_energy_mj)
                goodputs_gbps.append(goodput_gbps)

            row = {
                "Collective": coll_name,
                "Library": "nccl",
                "Protocol": prot,
                "Algorithm": alg, # Assuming scaleout describes the algorithm
                "Channels": ch, # Assuming scaleout describes the algorithm
                "Threads": th, # Assuming scaleout describes the algorithm
                "Data type": data_type,
                "Message size (B)": msg_size,
                "Num. nodes": num_nodes,
                "Num. GPUs": num_gpus,
                "GPUs per node": num_gpus / num_nodes,
                "Time (ms)": sum(times_ms)/len(times_ms),
                "Device Energy (mJ)":  sum(dev_energies_mj)/len(dev_energies_mj),
                "Host Energy (mJ)": sum(df_msg[host_energy_col].values)/len(runs),
                "Goodput (Gb/s)": sum(goodputs_gbps)/len(goodputs_gbps),
                "Power Trace (uw)": median_tuples(power_traces)

            }
            
            rows.append(row)


        # ------------------------------------
        # DO SOMETHING HERE
        # Example placeholders:
        #
        # avg_latency = df_run["latency"].mean()
        # max_bw = df_run["bandwidth"].max()
        #
        # or accumulate results in a list
        # ------------------------------------

   
    return rows


def main():
    parser = argparse.ArgumentParser(description="Parse oneCCL CSV results")
    parser.add_argument("--in-csv-folder", required=True, help="Input CSV folder")
    parser.add_argument("--out-csv-folder", required=True, help="Output CSV folder")

    args = parser.parse_args()

    in_folder = Path(args.in_csv_folder)
    out_folder = Path(args.out_csv_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    
    all_rows = [] # Contains all parsed rows from all CSVs for differnt scalein and scaleout
    for csv_file in in_folder.glob("*.csv"):
        rows = process_csv(csv_file, out_folder)  # returns list of dicts
        all_rows.extend(rows)                      # append to final list

    # Convert the full list of rows into a DataFrame
    df_final = pd.DataFrame(all_rows)
   
    # Collective name 
    folder_name = in_folder.name
    df_final.to_csv(out_folder / f"{folder_name}.csv", index=False)

if __name__ == "__main__":
    main()
