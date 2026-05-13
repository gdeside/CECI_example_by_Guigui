# Job Submission Helpers

Python scripts for **programmatically generating and submitting SLURM jobs** without writing sbatch files by hand. Useful when you need to submit many jobs with varying parameters.

## Files

| File | Description |
|------|-------------|
| `submit_job_helper_GPU.py` | Generates and submits a GPU sbatch script from Python |
| `submit_job_helper_R.py` | Generates and submits an R sbatch script from Python |

## How it works

Each helper builds an sbatch script as a string, writes it to a temporary file, and calls `subprocess.run(["sbatch", ...])` to submit it. You can loop over parameter lists to submit multiple jobs in one go.

## Example usage

```bash
python3 submit_job_helper_GPU.py
python3 submit_job_helper_R.py
```

Adapt the scripts by editing the resource parameters (CPUs, memory, time) and the command that gets executed inside the job.
