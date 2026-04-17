# Run a pbs script to execute the alltoall occl benchmark on aurora with different env variables
#!/usr/bin/env python3

import argparse
import subprocess
import os
import shutil
from pathlib import Path


# This script execute a2a with SYCL path by using compute kernels vs Level Zero path with compute eng. on single node.
# The goal is to compare the differences between transder data with computaiton kenrels vs copy engines on Intel GPUs.
# Here we are not setting differnt algorithms.
# Generic variable to enable the SYCL / LevelZero, Copy eng or Compute kernels and RDMA.

# Set SYCL path env variables to enable sycl kernels on single node and multi-node when are available.
sycl_kernel_env_vars = {"CCL_SYCL_COPY_ENGINE": "0", 
            "CCL_ENABLE_SYCL_KERNELS": "1", 
            "CCL_SYCL_KERNEL_COPY": "1", 
            "CCL_SYCL_ENABLE_DIRECT_GPU_RDMA" : "1", 
            "CCL_SYCL_ENABLE_PIPELINE_GPU_RDMA" : "1", 
            "CCL_SYCL_PIPELINE_GPU_RDMA" : "1",
            "CCL_SYCL_AUTO_USE_TMP_BUF" : "0",
            "CCL_SYCL_FORCE_USE_TMP_BUF_SCALEOUT": "0",
            "CCL_TOPO_ALGO" : "1"
            }
# Set LevelZero path env variables to enable oneCCL execution using SPIR-V kernels with LevelZero or copy eng
lz_env_vars = {
            # "CCL_SYCL_COPY_ENGINE" : "1", 
            "CCL_ENABLE_SYCL_KERNELS" : "0", 
            "CCL_SYCL_KERNEL_COPY" : "0",
            "CCL_SYCL_ENABLE_DIRECT_GPU_RDMA" : "1", 
            "CCL_SYCL_ENABLE_PIPELINE_GPU_RDMA" : "1", 
            "CCL_SYCL_PIPELINE_GPU_RDMA" : "1",
            "CCL_SYCL_AUTO_USE_TMP_BUF" : "0",
            "CCL_TOPO_ALGO" : "1"}

# All reduce variable
ar_sycl_alg_env_vars = {"CCL_SYCL_ALLREDUCE_SCALEOUT": ""}

ar_lz_alg_env_vars = {"CCL_ALLREDUCE_SCALEOUT" : ""}

ar_sycl_kernel_env_vars = {"CCL_ALLREDUCE": "topo"
                        #    , "CCL_REDUCE_SCATTER_MONOLITHIC_KERNEL": "1"
                        }
ar_lz_copy_env_vars = {"CCL_ALLREDUCE": "topo", "CCL_REDUCE_SCATTER_MONOLITHIC_KERNEL": "0"}
ar_lz_kernel_env_vars = {"CCL_ALLREDUCE": "topo", "CCL_REDUCE_SCATTER_MONOLITHIC_KERNEL": "1"}

# AllToAll variable
a2a_alg_env_vars = {"CCL_ALLTOALLV_SCALEOUT" : "",
           "CCL_ALLTOALL_SCALEOUT" : ""}
a2a_sycl_kernel_env_vars = {"CCL_ALLTOALL": "topo"}
a2a_lz_copy_env_vars = {"CCL_ALLTOALL": "topo", "CCL_ALLTOALLV_MONOLITHIC_KERNEL": "0", "CCL_ALLTOALLV_MONOLITHIC_READ_KERNEL": "0"}
a2a_lz_kernel_env_vars = {"CCL_ALLTOALL": "topo", "CCL_ALLTOALLV_MONOLITHIC_KERNEL": "1", "CCL_ALLTOALLV_MONOLITHIC_READ_KERNEL": "1"}



# Map collective → algorithms
#TODO: alltoall do not have a SYCL path only all reduce have it.
algorithms_map = {
    # "a2a": ["direct", "naive", "scatter"],
    "ar_lz": ["direct", "nreduce", "rabenseifner", "ring", "double_tree", "recursive_doubling"],
    "ar_sycl": ["direct", "auto", "rabenseifner", "ring"],
    "a2a": ["naive", "scatter"] 
}

scaleup_modes = [0, 1] # 0: use copy engines, 1: use GPU compute kernel
# scaleup_modes = [1] # 0: use copy engines, 1: use GPU compute kernel

def set_env(pbs_file_to_modify, coll, coll_path, num_nodes):
    path_name = coll_path.lower()
    if path_name == "sycl_kernel":
        selected_env_vars = dict(sycl_kernel_env_vars)
        if coll == "a2a":
            selected_env_vars.update(a2a_sycl_kernel_env_vars)
        elif coll == "ar":
            selected_env_vars.update(ar_sycl_kernel_env_vars)
        else:
            raise ValueError(f"Unknown collective: {coll}")
    elif path_name == "lz_copy":
        selected_env_vars = dict(lz_env_vars)
        if coll == "a2a":
            selected_env_vars.update(a2a_lz_copy_env_vars)
        elif coll == "ar":
            selected_env_vars.update(ar_lz_copy_env_vars)
        else:
            raise ValueError(f"Unknown collective: {coll}")
    elif path_name == "lz_kernel":
        selected_env_vars = dict(lz_env_vars)
        if coll == "a2a":
            selected_env_vars.update(a2a_lz_kernel_env_vars)
        elif coll == "ar":
            selected_env_vars.update(ar_lz_kernel_env_vars)
        else:
            raise ValueError(f"Unknown collective: {coll}")
    else:
        raise ValueError(f"Unknown collective path: {coll_path}")

    
    
    if int(num_nodes) > 1:
        if coll == "ar":
            if path_name == "sycl_kernel":
                selected_env_vars.update(ar_sycl_alg_env_vars)
            elif path_name == "lz_copy" or path_name == "lz_kernel":   
                selected_env_vars.update(ar_lz_alg_env_vars)  
            else:
                raise ValueError(f"Unknown collective path: {coll_path}. Speicify sycl_kernel, lz_copy or lz_kernel")
        elif coll == "a2a" :
            selected_env_vars.update(a2a_alg_env_vars)
        else:
            raise ValueError(f"Unknown collective: {coll}")
            
    with open(pbs_file_to_modify, 'r') as f:
        lines = f.readlines()

    with open(pbs_file_to_modify, 'w') as f:
        inserted = False
        for line in lines:
            f.write(line)

            if not inserted and "CCL_LOG_LEVEL" in line:
                for env_var, val in selected_env_vars.items():
                    f.write(f"export {env_var}={val}\n")
                inserted = True



def set_pbs(pbs_file_to_modify, log_dir, num_nodes, coll, partition, pbs_time, coll_path, alg):
    # Update the PBS script with nodes, logs, and job name
    with open(pbs_file_to_modify, 'r') as f:
        lines = f.readlines()
    
    base_name = f"{coll}_n{num_nodes}_{coll_path}"
    out_path = os.path.join(log_dir, "pbs-out", f"{base_name}_{num_nodes}_{alg}_{coll_path}_{partition}.out")
    err_path = os.path.join(log_dir, "pbs-out", f"{base_name}_{num_nodes}_{alg}_{coll_path}_{partition}.err")
    
    with open(pbs_file_to_modify, 'w') as f:
        for line in lines:
            if line.startswith("#PBS -l select="):
                f.write(f"#PBS -l select={num_nodes}\n")
            elif line.startswith("#PBS -N"):
                if int(num_nodes) > 1:
                    f.write(f"#PBS -N {coll}_n{num_nodes}_{coll_path}_{alg}\n")
                else:
                    f.write(f"#PBS -N {coll}_n{num_nodes}_{coll_path}\n")
            elif line.startswith("#PBS -o"):
                f.write(f"#PBS -o {out_path}\n")
            elif line.startswith("#PBS -e"):
                f.write(f"#PBS -e {err_path}\n")
            elif line.startswith("#PBS -q"):
                f.write(f"#PBS -q {partition}\n")
            elif line.startswith("#PBS -l walltime="):
                f.write(f"#PBS -l walltime={pbs_time}\n")
            else:
                f.write(line)

def str2bool(v):
    return v.lower() in ("true", "1", "yes")

def main():
    parser = argparse.ArgumentParser(description="Submit PBS jobs for collectives")
    parser.add_argument("--pbs", required=True, help="Path to PBS script template")
    parser.add_argument("--exe", required=True, help="Path to executable")
    parser.add_argument("--log", required=True, help="Path to the log directory")
    parser.add_argument("--env", required=True, help="Patht to the script file to set the environment")
    parser.add_argument("--partition", required=True, default="debug", help="Specify the partition on the target cluster")
    parser.add_argument("--time", required=True, default="00:10:00", help="Max timeout of the PBS experiment: HH:MM:SS")
    parser.add_argument("--path", required=True, default="00:10:00", help="Specify the oneCCL collective path: sycl_kernel, lz_kernel or lz_copy")
    parser.add_argument("--coll", required=True, default="00:10:00", help="Specify the oneCCL collective")
    parser.add_argument("--nodes", required=True, default="00:10:00", help="Specify the number of nodes")
    parser.add_argument("--scaleout-alg", required=False, default="00:10:00", help="Specify the scaleup algorithm to use for allreduce or alltoall")
    
    
        


    args = parser.parse_args()
    
    if args.coll != "a2a" and args.coll != "ar":
        raise ValueError(f"Unknown collective: {args.coll}. Use --coll=a2a or --coll=ar")
    
    if args.coll == "ar" and args.path == "sycl_kernel":
        avail_coll_alg = algorithms_map["ar_sycl"]
    elif args.coll == "ar" and (args.path == "lz_copy" or args.path == "lz_kernel"):
        avail_coll_alg = algorithms_map["ar_lz"]
    else:
        avail_coll_alg = algorithms_map[args.coll]

    
    if args.scaleout_alg not in avail_coll_alg:
        raise ValueError(f"Unknown alg for {args.coll}. Use --scaleout-alg to specify a valid algorithm among {avail_coll_alg}")
    else:
        for key in a2a_alg_env_vars:
            a2a_alg_env_vars[key] = args.scaleout_alg
        if args.path == "sycl_kernel":
            for key in ar_sycl_alg_env_vars:
                ar_sycl_alg_env_vars[key] = args.scaleout_alg   
        elif args.path == "lz_kernel" or args.path == "lz_copy":
            for key in ar_sycl_alg_env_vars:
                ar_sycl_alg_env_vars[key] = args.scaleout_alg
        else:
            raise ValueError(f"Unknown path {args.path}. Specify sycl_kernel, lz_kernel or lz_copy")
    
    # Create directory for logs
    os.makedirs(args.log, exist_ok=True)
    os.makedirs(os.path.join(args.log, "pbs-out"), exist_ok=True)

    occl_path=""
    comm_type=""
    # Detect the type of path SYCL or LZ.
    if args.path == "sycl_kernel":      
        occl_path = "SYCL"
        comm_type = "kernel"
    elif args.path == "lz_kernel":
        occl_path = "LZ"
        comm_type = "kernel"
    elif args.path == "lz_copy":
        occl_path = "LZ"
        comm_type = "copy"
    else:
        raise ValueError(f"Unknown path: {args.path}. Must be 'sycl', 'lz_kernel' or 'lz_copy'.")

    num_nodes = int(args.nodes)
    
    # --- New logic for generating PBS files ---
    pbs_template_path = Path(args.pbs)
    pbs_dir = pbs_template_path.parent
    generated_dir = pbs_dir / "generated"

    # Clear existing generated files if the directory exists
    if generated_dir.exists():
        shutil.rmtree(generated_dir)
    generated_dir.mkdir(parents=True, exist_ok=True)

    # Define the path for the new PBS file
    generated_pbs_filename = f"{args.coll}_n{num_nodes}_{args.path}.sh"
    generated_pbs_path = generated_dir / generated_pbs_filename

    # Copy the template to the generated path
    shutil.copyfile(pbs_template_path, generated_pbs_path)
    
    # Update the generated PBS script with parameters
    set_pbs(str(generated_pbs_path), args.log, num_nodes, args.coll, args.partition, args.time, args.path, args.scaleout_alg)
    set_env(str(generated_pbs_path), args.coll, args.path, num_nodes)
    
    
    is_single_node = num_nodes == 1
    scaleup_mode = 1 if args.path == "sycl" else 0
    
    if is_single_node:
        submit_job(
            pbs=str(generated_pbs_path), # Use the generated PBS path
            exe=args.exe,
            csv_log=args.log,
            coll=args.coll,
            alg_name="Default",
            occl_path=occl_path,
            comm_type=comm_type,
            env=args.env
        )
    else:
        submit_job(
            pbs=str(generated_pbs_path), # Use the generated PBS path
            exe=args.exe,
            csv_log=args.log,
            coll=args.coll,
            alg_name=args.scaleout_alg,
            occl_path=occl_path,
            comm_type=comm_type,
            env=args.env
        )

def submit_job(pbs, exe, csv_log, coll, alg_name, occl_path, comm_type, env):
    env_vars = {
        "EXE_PATH": exe,
        "CSV_LOG_DIR": csv_log,
        "COLLECTIVE": coll,
        "ONECCL_PATH": occl_path,
        "SCALEOUT_ALG": alg_name,
        "COMM_TYPE": comm_type,
        "SET_ENV_PATH": env,
    }

    env_string = ",".join(f"{k}={v}" for k, v in env_vars.items())

    cmd = ["qsub", "-v", env_string, pbs]

    print("Submitting:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
