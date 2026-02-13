
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import os
import itertools
import re
import math


collective_col="Collective"
lib_col="Library"
data_type_col="Data type"
msg_size_col="Message size (B)"
num_nodes_col="Num. nodes" # Total number of node 
num_gpus_col= "Num. GPUs" # Total number of GPUs
gpu_per_node_col="GPUs per node"
prot_col="Protocol"
alg_col="Algorithm"
ch_col="Channels"
th_col="Threads" 
time_col="Time (ms)" # max time between all ranks
device_energy_col= "Device Energy (mJ)" # sum of all devices
host_energy_col= "Host Energy (mJ)"
goodput_col="Goodput (Gb/s)"
power_trace_col="Power Trace (uw)"

collectives=["a2a"]


# def parse_power_trace(s):
#     matches = re.findall(r'\(([^;]+);([^)]+)\)', s)
#     return [(float(t), int(v)) for t, v in matches]
def parse_power_trace(s):
    print(s)
    if not isinstance(s, str):
        return []

    # Remove surrounding brackets if present
    s = s.strip().strip("[]")
    # Extract (time;value) pairs
    matches = re.findall(r'\((\d+);(\d+)\)', s)

    # Convert to numeric types
    return [(int(t), int(v)) for t, v in matches]

def generate_plot(df, out_dir, collective_name):
    msg_size = df[msg_size_col].values[0]
    os.makedirs(out_dir, exist_ok=True)

    # -----------------------------
    # Convert traces into long format
    # -----------------------------
    records = []
    for _, row in df.iterrows():
        channel = row[ch_col]
        protocol = row[prot_col]
        print(protocol)
        trace = parse_power_trace(row[power_trace_col])

        for t, power in trace:
            records.append({
                "Time": t,
                "Power": power/1e6,
                "Channel": channel,
                "Protocol": protocol
            })

    if not records:
        print("No valid traces found.")
        return

    plot_df = pd.DataFrame(records)

    # Improve seaborn style
    sns.set_theme()

    # =====================================================
    # 1️⃣ SUBPLOT PER CHANNEL (hue = Protocol)
    # =====================================================
    channels = sorted(plot_df["Channel"].unique())
    n_channels = len(channels)

    fig1, axes1 = plt.subplots(
        1, n_channels,
        figsize=(6 * n_channels, 5),
        sharey=True
    )

    if n_channels == 1:
        axes1 = [axes1]

    for ax, channel in zip(axes1, channels):
        df_channel = plot_df[plot_df["Channel"] == channel]

        sns.lineplot(
            data=df_channel,
            x="Time",
            y="Power",
            hue="Protocol",
            ax=ax
        )

        ax.set_title(f"Channel {channel}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Power")
    num_nodes = df[num_nodes_col].values[0]
    fig1.suptitle(f"{collective_name} - Power Trace per Channel")
    fig1.tight_layout()
    fig1.savefig(os.path.join(out_dir, f"power_per_channel_nodes{num_nodes}_size{msg_size}.pdf"))
    plt.close(fig1)

    # =====================================================
    # 2️⃣ SUBPLOT PER PROTOCOL (hue = Channel)
    # =====================================================
    protocols = sorted(plot_df["Protocol"].unique())
    n_protocols = len(protocols)

    fig2, axes2 = plt.subplots(
        1, n_protocols,
        figsize=(6 * n_protocols, 5),
        sharey=True
    )

    if n_protocols == 1:
        axes2 = [axes2]

    for ax, protocol in zip(axes2, protocols):
        df_protocol = plot_df[plot_df["Protocol"] == protocol]

        sns.lineplot(
            data=df_protocol,
            x="Time",
            y="Power",
            hue="Channel",
            ax=ax
        )

        ax.set_title(f"Protocol {protocol}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Power")

    fig2.suptitle(f"{collective_name} - Power Trace per Protocol")
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, f"power_per_protocol_nodes{num_nodes}_size{msg_size}.pdf"))
    plt.close(fig2)

def main():
    parser = argparse.ArgumentParser(description="NCCL energy characterization with different parameters")
    parser.add_argument('--csv-file', type=str, required=True, help="CSV file containing host/device energy and perforamnce for each  library collective and tuning parameter. e.g nccl, ar, nthreads")
    parser.add_argument('--out-dir', type=str, required=True, help="path to the plot directory. The script generate in this directory a new folder for each collective")
    
    args = parser.parse_args()
    csv_file = Path(args.csv_file)
    out_dir = Path(args.out_dir)

        
    all_dfs = pd.read_csv(csv_file)
    nodes = [1, 2, 4, 8]
    
    for msg_size in all_dfs[msg_size_col].unique():
        df_msg = all_dfs[all_dfs[msg_size_col]==msg_size]
        for node in nodes:
            df_node = df_msg[df_msg[num_nodes_col]==node]
            for coll in collectives:
                df_coll=df_node[df_node["Collective"]==coll]
                if df_coll.empty:
                    print(f"No data for {coll} collective")
                    continue
                generate_plot(df_coll, f"{out_dir}/{coll}/", coll)

    
if __name__ == "__main__":
    main()