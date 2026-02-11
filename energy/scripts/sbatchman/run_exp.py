from pathlib import Path
import argparse
import sbatchman as sbm


def main():
    parser = argparse.ArgumentParser(
        description="Launch sbatchman jobs from a YAML file."
    )
    parser.add_argument(
        "--jobs-file",
        type=Path,
        required=True,
        help="Path to the YAML file containing job definitions."
    )
    args = parser.parse_args()
    jobs_file = args.jobs_file

    try:
        launched_jobs = sbm.launch_jobs_from_file(jobs_file)
        print(f"Successfully launched {len(launched_jobs)} jobs.")
        for job in launched_jobs:
            print(f"  - Job ID: {job.job_id}, Config: {job.config_name}, Tag: {job.tag}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
