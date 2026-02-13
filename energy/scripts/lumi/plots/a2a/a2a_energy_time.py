# x-axis: GB/s
# y-axis: Energy J
# each point is a different NTHREADS 64, 128, 256, 512
# I have a plot with one row and 4 columns each subplot shows the data for a different size.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import os
import itertools
import re
import math
from paretoset import paretoset

sns.set_theme()

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

# Start with some standard markers that are filled
base_markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']

# Generate regular polygons and stars with varying rotation
# (num_sides, style, rotation)
poly_markers = []
num_sides_options = [4, 6, 7, 8]
styles = [1]  # 0 = regular polygon, 1 = star
rotations = [0,45]

for rotation in rotations:
    for ns in num_sides_options:
        for style in styles:
            poly_markers.append((ns, style, rotation))
            if len(poly_markers) + len(base_markers) >= 20:
                break  # break style loop early if reached 20 markers
        if len(poly_markers) + len(base_markers) >= 20:
            break  # break num_sides_options loop early if reached 20 markers

# Combine all markers into one list
all_markers = base_markers + poly_markers[:20 - len(base_markers)]

tuning_parameters_map={
    "nchannels": "NCCL_MIN/MAX_CTAS",
    "nthreads": "NCCL_NTHREADS",
    "launch": "NCCL_LAUNCH_MODE",
    "buffsize":"NCCL_BUFFSIZE"
}

# collectives=["a2a"]
collectives=["ar", "a2a"]



libraries=["rccl"]


# from paretoset import paretoset
byte_mapping = {
    1: "1 B",
    2: "2 B",
    4: "4 B",
    8: "8 B",
    16: "16 B",
    32: "32 B", 
    64: "64 B",
    128: "128 B",
    256: "256 B",
    512: "512 B",
    1024: "1024 B",
    2048: "2048 B",
    4096: "4 KiB",
    8192: "8 KiB",
    16384: "16 KiB",
    32768: "32 KiB",
    65536: "64 KiB",
    131072: "128 KiB",
    262144: "256 KiB",
    524288: "512 KiB",
    1048576:"1024 KiB",
    2097152: "2 MiB",
    4194304:"4 MiB",
    8388608: "8 MiB",
    16777216: "16 MiB",
    33554432: "32 MiB",
    67108864: "64MiB",
    134217728: "128MiB",
    268435456: "256MiB",
    536870912: "512 MiB",
    1073741824: "1GiB"
}


def print_pareto(df, x_obj, y_obj, pareto_line_color, pareto_label_name, ax):
    # Compute the pareto set point and print on the plot
    # Creaete a data frame with energy and speedup
    df_xobj_yobj = pd.DataFrame({f"{x_obj}": df[x_obj], f"{y_obj}": df[y_obj]})
    mask = paretoset(df_xobj_yobj, sense=["max", "min"])
    pset = df_xobj_yobj[mask]
    pset = pset.sort_values(by=[f"{x_obj}"])
    df_filtered = df.merge(pset, on=[f"{x_obj}", f"{y_obj}"])

    
    ############# PRINT PARETO FRONT WITH A RED LINE ################
    
    np_array = pset.to_numpy()
    pset_size = len(pset[f"{x_obj}"])
    
    # cur_xlim_left, cur_xlim_right = plt.xlim()
    # cur_xlim_bottom, cur_ylim_top = plt.ylim()
    # x1, y1 = [cur_xlim_left, np_array[0][0]], [np_array[0][1], np_array[0][1]]
    # plt.plot(x1, 0, color=pareto_line_color, linewidth=2.5, label="Pareto-front")
   
    for i in range(pset_size):
        if not (i == pset_size-1):
            current_x = np_array[i][0]
            current_y = np_array[i][1]
            next_x = np_array[i+1][0]
            next_y = np_array[i+1][1]
            # Horizontal line: from current to (next_x, current_y)
            # x_h = [current_x, next_x]
            # y_h = [current_y, current_y]
            # x_v = [next_x, next_x]
            # y_v = [current_y, next_y]
            # # Plot both segments
            # ax.plot(x_h, y_h, color=pareto_line_color, linewidth=2.5)  # horizontal segment
            # ax.plot(x_v, y_v, color=pareto_line_color, linewidth=2.5)  # vertical segment
            x1, y1 = [current_x, current_x], [current_y, next_y]
            x2, y2 = [current_x, next_x], [next_y, next_y]
            ax.plot(x2, y2,x1, y1, color=pareto_line_color, linewidth=2.5)
    
    ############# PRINT PARETO FRONT WITH A RED LINE ################
    return df_filtered



def generate_plot(df, out_dir, app, norm_energy, var_param, perf_loss, energy_saving):
    os.makedirs(out_dir, exist_ok=True)
    
    
    # Ensure both columns have the same type
    df[msg_size_col] = df[msg_size_col].astype(int)
   
    
    
    # I need the same color and marker for all the configuration so I need to create a map
   
    ################### GENERATE COLO AND MARKER MAP #########################
    
       # type can be host, device or host_device
    # param_filter define the paramter fixed for filtering the results
    def host_device_energy_plot(df, type):
        # Unique message sizes
        msg_sizes = sorted(df[msg_size_col].unique())
        n_sizes = len(msg_sizes)
        # Grid size: 3 columns, ceil(n_sizes / 3) rows
        ncols = 3
        nrows = math.ceil(n_sizes / ncols)
        
        data_types = sorted(df[data_type_col].unique())
        for data_t in data_types:
            print(f'Data type: {data_t}')
            num_nodes=df[num_nodes_col].values[0]
            plot_name = f"rccl_{app}_{type}_{data_t}_{num_nodes}.pdf"
            df_t = df[df[data_type_col]==data_t]
            # Create subplots
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 4), squeeze=False)

            for i, msg_size in enumerate(msg_sizes):
                if int(msg_size) == 167777216:
                    continue
                row = i // ncols
                col = i % ncols
                ax = axes[row][col]
                filtered_data = df_t[df_t[msg_size_col]==msg_sizes[i]]
                
                if filtered_data.empty:
                    continue
                
                x_obj="Min Goodput (Gb/s)"
                y_obj="Device Energy (J)"
                filtered_data.loc[:, "Device Energy (J)"] = filtered_data[device_energy_col] / 1_000 # mJ to J
                if norm_energy:
                    filtered_data["Device Energy per Second (J/s)"] = filtered_data["Device Energy (J)"] / (filtered_data[time_col] / 1_000)
                    # filtered_data["Host Energy per Second (J/s)"] = filtered_data["Host Energy (J)"] / (filtered_data[time_col] / 1_000)

                if "device" == type:
                    if norm_energy:
                        y_obj="Device Energy per Second (J/s)"
                    else:
                        y_obj="Device Energy (J)"
                elif "host" == type: 
                    if norm_energy == True:
                        y_obj="Host Energy per Second (J/s)"
                    else:
                        y_obj="Host Energy (J)"
                    
                    # filtered_data = filtered_data[filtered_data[y_obj] >= 0]
                else:
                    if norm_energy == True:
                        y_obj="Host and Device Energy per Second (J/s)"
                        filtered_data[y_obj]=filtered_data["Host Energy per Second (J/s)"] + filtered_data["Device Energy per Second (J/s)"]
                    else:
                        y_obj="Host and Device Energy (J)"
                        filtered_data[y_obj]=filtered_data["Host Energy (J)"] + filtered_data["Device Energy (J)"]
                
                filtered_data.loc[:, x_obj] = filtered_data[goodput_col]
                
                
                # For readibility we use different unit measure for each plot
                if i < 6:
                    x_obj="Min Goodput Mb/s"
                    filtered_data[x_obj] = (filtered_data[goodput_col] * 1_000)
                
                # filtered_data[x_obj] = filtered_data[x_obj].round(3) # Uncomment this line to have the Gb/s
                x_obj="Time [ms]"
                filtered_data[x_obj] = filtered_data[time_col].round(3) # Uncomment this line to have the Gb/s
                
                
                
                # pset = print_pareto(filtered_data, x_obj, y_obj, "red", "Pareto-front", ax)
                print(filtered_data)
                scatter = sns.scatterplot(
                    data=filtered_data,
                    x=x_obj,
                    y=y_obj,
                    hue=prot_col,
                    style=ch_col,
                    palette="tab20",
                    ax=ax,
                    s=150,
                    edgecolor="black",
                )
                
                ax.set_title(f"Size {byte_mapping[int(msg_size)]}")
                ax.legend(fontsize='x-small', title_fontsize='small')
            plt.tight_layout()
            # plt.subplots_adjust(right=0.83)  # Leave space for legend
            # Construct the full path
            dir_path = f"{out_dir}/{type}"

            # Create the directory if it doesn't exist
            os.makedirs(dir_path, exist_ok=True)

            plt.savefig(f"{dir_path}/{plot_name}")
            plt.clf()      # Clears the current figure
            plt.cla()      # Clears the current axes
            plt.close()   

    host_device_energy_plot(df, f"device")
                

    # ############ START PRINT PLOT FOR ALL PARAMS #########
    # # Print all the configurations in one plot when var_param is all
    # if(var_param == "all"):
    #     # all configurations
    #     print("Printing HOST energy ALL")
    #     host_device_energy_plot(df, "host", "all")
    #     print("Printing DEVICE energy ALL")
    #     host_device_energy_plot(df, "device", "all")
    #     print("Printing HOST_DEVICE energy ALL")
    #     host_device_energy_plot(df, "host_device",  "all")
        
    #     # print only pareto optimal solution
    #     print("Printing HOST energy ALL_PARETO")
    #     host_device_energy_plot(df, "host", "all_pareto")
    #     print("Printing DEVICE energy ALL_PARETO")
    #     host_device_energy_plot(df, "device", "all_pareto")
    #     print("Printing HOST_DEVICE energy ALL_PARETO")
    #     host_device_energy_plot(df, "host_device",  "all_pareto")
    #     # ############ END PRINT PLOT FOR ALL PARAMS #########
    # else:
    #     # alg, prot, nthreads, nchannels 
    #     params = ['threads', 'alg', 'prot', 'channels']
    #     params.remove(var_param)
    #     for x in sorted(df[params[0]].unique()):
    #         for y in sorted(df[params[1]].unique()):
    #             for z in sorted(df[params[2]].unique()):
    #                 print(f"Params: {params[0]}-{x}, {params[1]}-{y}, {params[2]}-{z}")        
    #                 filtered_df = df[
    #                                 (df[params[0]] == x) &
    #                                 (df[params[1]] == y) &
    #                                 (df[params[2]] == z)
    #                             ]
    #                 host_device_energy_plot(filtered_df, f"host", f"{params[0]}{x}_{params[1]}{y}_{params[2]}{z}_{var_param}")
    #                 host_device_energy_plot(filtered_df, f"device", f"{params[0]}{x}_{params[1]}{y}_{params[2]}{z}_{var_param}")
    #                 host_device_energy_plot(filtered_df, f"host_device",f"{params[0]}{x}_{params[1]}{y}_{params[2]}{z}_{var_param}")
                

    ############ START PRINT PLOT FOR EACH TUNING PARAMETER ###
    
    # channels = sorted(df["channels"].unique())
    # # Different plot for each nchannels value
    # for channel in channels:
    #     # take only the row with channels == channel
    #     filtered_df = df[df['channels']==channel]
    #     host_device_energy_plot(filtered_df, "host", f"channels{channel}")
    #     host_device_energy_plot(filtered_df, "device", f"channels{channel}")
    #     host_device_energy_plot(filtered_df, "host_device",  f"channels{channel}")
        
    # threads = sorted(df["threads"].unique())
    # # Different plot for each nthreads value
    # for t in threads:
    #     # take only the row with threads == t
    #     filtered_df = df[df['threads']==t]
    #     host_device_energy_plot(filtered_df, "host", f"threads{t}")
    #     host_device_energy_plot(filtered_df, "device", f"threads{t}")
    #     host_device_energy_plot(filtered_df, "host_device", f"threads{t}")
    
    # # Different plot for each algorithm
    # algorithms = sorted(df['alg'].unique())
    # for alg in algorithms:
    #     # take only the row with algorithm == alg
    #     filtered_df = df[df['alg']==alg]
    #     host_device_energy_plot(filtered_df, "host", f"alg{alg}")
    #     host_device_energy_plot(filtered_df, "device", f"alg{alg}")
    #     host_device_energy_plot(filtered_df, "host_device", f"alg{alg}")
    
    # # Different plot for each protcol
    # protocols = sorted(df['prot'].unique())
    # for prot in protocols:
    #     # take only the row with threads == t
    #     filtered_df = df[df['prot']==prot]
    #     host_device_energy_plot(filtered_df, "host", f"prot{prot}")
    #     host_device_energy_plot(filtered_df, "device", f"prot{prot}")
    #     host_device_energy_plot(filtered_df, "host_device", f"prot{prot}")
    
    ########## END PRINT PLOT FOR EACH TUNING PARAMETER ##########

    
    
def main():
    parser = argparse.ArgumentParser(description="NCCL energy characterization with different parameters")
    parser.add_argument('--csv-file', type=str, required=True, help="CSV file containing host/device energy and perforamnce for each  library collective and tuning parameter. e.g nccl, ar, nthreads")
    parser.add_argument('--out-dir', type=str, required=True, help="path to the plot directory. The script generate in this directory a new folder for each collective")
    parser.add_argument('--norm-energy', type=str, default="False", help="True to normalize the energy with the time (J/s)")
    # Given the var-param I will generate different plot for each different value of var parameter by fixing the other ones.
    parser.add_argument('--var-param', type=str, default="nthreads", help="tuning parameter to vary. e.g algorithm, protocol, nthreads, nchannels")
    parser.add_argument('--perf-loss', type=str, default="100", help="Print only the point with a performance loss of x% compared to the best performance configuration")
    parser.add_argument('--energy-saving', type=str, default="0", help="Print only the point with an energy saving >= x% compared to the point with the best performance")
    
    args = parser.parse_args()
    csv_file = Path(args.csv_file)
    out_dir = Path(args.out_dir)
    norm_energy = (args.norm_energy=="True")
    var_param = args.var_param
    perf_loss = int(args.perf_loss)
    energy_saving = int(args.energy_saving)    

    if(norm_energy==False):
        print("Energy is not normalized")
    else:
        print("Energy is normalized")
        
    all_dfs = pd.read_csv(csv_file)
    nodes = [1, 2, 4, 8]
    for node in nodes:
        df_node = all_dfs[all_dfs[num_nodes_col]==node]
        for coll in collectives:
            df_coll=df_node[df_node["Collective"]==coll]
            if df_coll.empty:
                print(f"No data for {coll} collective")
                continue
        generate_plot(df_coll, f"{out_dir}/{coll}/", coll, norm_energy, var_param, perf_loss, energy_saving)

    
if __name__ == "__main__":
    main()
