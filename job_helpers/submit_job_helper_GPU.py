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
    script_content += f"#SBATCH --partition={job_info['partition']}\n"
    script_content += f"#SBATCH --gres=gpu:{job_info['gpu']}\n"
    script_content += f"#SBATCH --cpus-per-task={job_info['cpus_per_task']}\n"
    script_content += f"#SBATCH --mem={job_info['mem']}\n"
    script_content += f"#SBATCH --time={job_info['time']}\n"
    script_content += f"#SBATCH --output={job_info['output']}\n\n"

    # Append the job script command
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

# Example usage for your GPU job
job_info = {
    "job_name": "Example_GPU",
    "partition": "gpu",
    "gpu": 1,
    "cpus_per_task": 4,
    "mem": "16G",
    "time": "00:30:00",
    "output": "example_GPU.txt",
    "script": "python3 bayesian_gpu_demo.py"
}

sbatch_script_path = "example_GPU.sbatch"
submit_job(job_info, sbatch_script_path)