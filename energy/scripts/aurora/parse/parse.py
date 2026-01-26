import argparse
from pathlib import Path
import pandas as pd
import re

msg_col="message_size_bytes"
run_col="run_id"
device_energy_col="device_energy_mj_1coll"
host_energy_col="host_energy_mj_1coll" 
goodput_col="goodput_Gb_per_s"
time_col="time_ms_1coll"

header_cols = [
    "Collective",
    "Library",
    "Scaleup",
    "Algorithm",
    "Data type",
    "Message size (B)",
    "Num. nodes", # Total number of nodes
    "Num. GPUs", # Total number of GPUs
    "GPUs per node", 
    "Time (ms)", # max time between all ranks
    "Device Energy (mJ)", # sum of all devices
    "Host Energy (mJ)",
    "Goodput (Gb/s)"
]

def parse_power_trace(s):
    matches = re.findall(r'\(([^;]+);([^)]+)\)', s)
    return [(t, int(v)) for t, v in matches]


def parse_filename(filename: str):
    """
    Expected format: collName_scaleup_scaleout.csv
    Example: allreduce_1_ring.csv
    """
    stem = Path(filename).stem
    parts = stem.split("_")

    if len(parts) < 3:
        raise ValueError(f"Invalid filename format: {filename}")

    coll_name = "_".join(parts[:-2])
    scaleup = int(parts[-2])
    scaleout = parts[-1]

    return coll_name, scaleup, scaleout


def translate_scaleup(scaleup: int):
    if scaleup == 1:
        return "kernelGPU"
    else:
        return "copyEng"

def process_csv(csv_path: Path, out_folder: Path):
    coll_name, scaleup, scaleout = parse_filename(csv_path.name)

    df = pd.read_csv(csv_path)
    num_nodes = df["total_nodes"].values[0]
    num_gpus = df["num_ranks"].values[0]
    
    # Add metadata columns
    df["scaleup"] = translate_scaleup(scaleup)
    df["scaleout"] = scaleout
    
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

                # Extract relevant metrics for this run
                time_ms = max(df_run[time_col].values)
                dev_energy_mj = sum(df_run[device_energy_col].values)
                goodput_gbps = min(df_run[goodput_col]) 
                
                times_ms.append(time_ms)
                dev_energies_mj.append(dev_energy_mj)
                goodputs_gbps.append(goodput_gbps)
            
            row = {
                "Collective": coll_name,
                "Library": "occl",
                "Scaleup": scaleup,
                "Algorithm": scaleout, # Assuming scaleout describes the algorithm
                "Data type": data_type,
                "Message size (B)": msg_size,
                "Num. nodes": num_nodes,
                "Num. GPUs": num_gpus,
                "GPUs per node": num_gpus / num_nodes,
                "Time (ms)": sum(times_ms)/len(times_ms),
                "Device Energy (mJ)":  sum(dev_energies_mj)/len(dev_energies_mj),
                "Host Energy (mJ)": sum(df_msg[host_energy_col].values)/len(runs),
                "Goodput (Gb/s)": sum(goodputs_gbps)/len(goodputs_gbps)
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

    # final_df = pd.DataFrame(rows, columns=header_cols)
    # # Save processed CSV
    # out_path = out_folder / csv_path.name
    # final_df.to_csv(out_path, index=False)
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
