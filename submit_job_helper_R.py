import datetime
import subprocess

def submit_job(job_info, sbatch_script_path):
    """
    Write and submit a job to the Slurm Workload Manager.

    :param job_info: Dictionary with job configuration parameters.
    :param sbatch_script_path: Path to save the generated sbatch script.
    :return job_id: The id of the submitted job.
    """
    # Initialize sbatch script content
    script_content = "#!/bin/bash\n"

    # Required SBATCH parameters
    script_content += f"#SBATCH --job-name={job_info['job_name']}\n"
    script_content += f"#SBATCH --time={job_info['time']}\n"
    script_content += f"#SBATCH --ntasks={job_info['ntasks']}\n"
    script_content += f"#SBATCH --cpus-per-task={job_info['cpus_per_task']}\n"
    script_content += f"#SBATCH --mem-per-cpu={job_info['mem_per_cpu']}\n"
    script_content += f"#SBATCH --mail-user={job_info['mail_user']}\n"
    script_content += f"#SBATCH --mail-type=FAIL\n"
    script_content += f"#SBATCH --mail-type=END\n"
    script_content += f"#SBATCH --output={job_info['output']}\n\n"

    # Append module load and script command
    script_content += f"module load {job_info['module']}\n\n"
    script_content += f"{job_info['script']}"

    # Write sbatch script to file
    with open(sbatch_script_path, 'w') as script_file:
        script_file.write(script_content)

    print(f"[INFO] {datetime.datetime.now().strftime('%d.%b %Y %H:%M:%S')}: Generated sbatch script: '{sbatch_script_path}'")

    # Submit the sbatch script and capture the job ID
    try:
        sbatch_output = subprocess.check_output(["sbatch", sbatch_script_path], stderr=subprocess.STDOUT)
        sbatch_output = sbatch_output.decode().strip()
        job_id = next(int(word) for word in sbatch_output.split() if word.isdigit())
        print(f"[INFO] Job submitted with ID: {job_id}")
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"ERROR: sbatch submission failed with error: {e.output.decode()}")
        raise e

# Example usage for your R job
job_info = {
    "job_name": "Example_R",
    "time": "00:30:00",
    "ntasks": 1,
    "cpus_per_task": 20,
    "mem_per_cpu": "4000",
    "mail_user": "guillaume.deside@uclouvain.be",
    "output": "example_R.txt",
    "module": "R",
    "script": "Rscript example_R.R"
}

sbatch_script_path = "example_R.sbatch"
submit_job(job_info, sbatch_script_path)