#!/bin/bash
#SBATCH --job-name=cv_project           # Job name
#SBATCH --output=cv_project_%j.out      # Standard output log (with job ID)
#SBATCH --error=cv_project_%j.err       # Standard error log (with job ID)
#SBATCH --partition=mtech                 # Partition name (adjust based on your cluster, e.g., gpu, compute)
#SBATCH --gres=gpu:1                    # Request 1 GPU (adjust if you need specific types like gpu:v100:1)
#SBATCH --cpus-per-task=4               # Number of CPU cores per task
#SBATCH --mem=16G                       # Job memory request
#SBATCH --time=12:00:00                 # Time limit hrs:min:sec

# Print some information about the job
echo "Job starting on $(hostname) at $(date)"
echo "Assigned GPU(s): $CUDA_VISIBLE_DEVICES"

# Change to the current directory where the script was submitted from
cd "$SLURM_SUBMIT_DIR"

# Ensure we are in the CV project directory
cd "CV project" 2>/dev/null || echo "Already inside CV project or directory not found"

# Activate the virtual environment
source venv/bin/activate

# Execute the script
# (You can modify the flags like --arch, --epochs below to match what you need)
python main.py --arch a --epochs 60 --mvtec data/mvtec --aitex data/aitex

echo "Job finished at $(date)"
