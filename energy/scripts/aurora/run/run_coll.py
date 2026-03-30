# Run a pbs script to execute the alltoall occl benchmark on aurora with different env variables
#!/usr/bin/env python3

import argparse
import subprocess
import os
import shutil
from pathlib import Path

# Map collective → algorithms
#TODO: allreduce ring algorith: reduce_scatter + allgather ring. Use CCL_RS_CHUNK_COUNT and CCL_RS_MIN_CHUNK_SIZE to control pipelining on reduce_scatter phase
algorithms_map = {
    # "a2a": ["direct", "naive", "scatter"],
    "ar": ["direct", "nreduce", "rabenseifner", "ring", "double_tree", "recursive_doubling"],
    "a2a": ["direct"], 
}

scaleup_modes = [0, 1] # 0: use copy engines, 1: use GPU compute kernel
# scaleup_modes = [1] # 0: use copy engines, 1: use GPU compute kernel

def set_pbs(pbs_file_to_modify, log_dir, num_nodes, coll, partition, pbs_time):
    # Update the PBS script with nodes, logs, and job name
    with open(pbs_file_to_modify, 'r') as f:
        lines = f.readlines()
    
    base_name = f"{coll}_n{num_nodes}"
    out_path = os.path.join(log_dir, "pbs-out", f"{base_name}.out")
    err_path = os.path.join(log_dir, "pbs-out", f"{base_name}.err")
    
    with open(pbs_file_to_modify, 'w') as f:
        for line in lines:
            if line.startswith("#PBS -l select="):
                f.write(f"#PBS -l select={num_nodes}\n")
            elif line.startswith("#PBS -N"):
                f.write(f"#PBS -N {coll}_n{num_nodes}\n")
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
    parser.add_argument("--coll", required=True, help="Collective name can be: a2a or ar")
    parser.add_argument("--nodes", required=True, help="Num. of nodes")
    parser.add_argument("--partition", required=True, default="debug", help="Specify the partition on the target cluster")
    parser.add_argument("--time", required=True, default="00:10:00", help="Max timeout of the PBS experiment: HH:MM:SS")


    args = parser.parse_args()

    # Create directory for logs
    os.makedirs(args.log, exist_ok=True)
    os.makedirs(os.path.join(args.log, "pbs-out"), exist_ok=True)


    if args.coll not in algorithms_map:
        raise ValueError(f"Unknown collective: {args.coll}")

    algorithms = algorithms_map[args.coll]

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
    generated_pbs_filename = f"{args.coll}_n{num_nodes}.sh"
    generated_pbs_path = generated_dir / generated_pbs_filename

    # Copy the template to the generated path
    shutil.copyfile(pbs_template_path, generated_pbs_path)
    
    # Update the generated PBS script with parameters
    set_pbs(str(generated_pbs_path), args.log, num_nodes, args.coll, args.partition, args.time)
    
    is_single_node = num_nodes == 1

    for scaleup_mode in scaleup_modes: # iterate over scaleup modes 
        if is_single_node:
            submit_job(
                pbs=str(generated_pbs_path), # Use the generated PBS path
                exe=args.exe,
                csv_log=args.log,
                coll=args.coll,
                scaleup_mode=scaleup_mode,
                alg_name="Default",
                env=args.env
            )
        else:
            for alg in algorithms:
                submit_job(
                    pbs=str(generated_pbs_path), # Use the generated PBS path
                    exe=args.exe,
                    csv_log=args.log,
                    coll=args.coll,
                    scaleup_mode=scaleup_mode,
                    alg_name=alg,
                    env=args.env
                )

def submit_job(pbs, exe, csv_log, coll, scaleup_mode, alg_name, env):
    env_vars = {
        "EXE_PATH": exe,
        "CSV_LOG_DIR": csv_log,
        "COLLECTIVE": coll,
        "SCALEUP_MODE": scaleup_mode,
        "SCALE_OUT": alg_name,
        "SET_ENV_PATH": env,
    }

    env_string = ",".join(f"{k}={v}" for k, v in env_vars.items())

    cmd = ["qsub", "-v", env_string, pbs]

    print("Submitting:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
