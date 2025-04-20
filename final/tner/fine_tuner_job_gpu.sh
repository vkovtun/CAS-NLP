#!/bin/bash
#SBATCH --job-name="TNER Fine Tuning"
#SBATCH --time=09:00:00
#SBATCH --mem-per-cpu=32GB
#SBATCH --cpus-per-task=2
#SBATCH --qos=job_gpu_preemptable
#_SBATCH --partition=gpu-invest
#_SBATCH --gres=gpu:rtx3090:1
#_SBATCH --gres=gpu:rtx4090:1
#_SBATCH --gres=gpu:a100:1
#SBATCH --gres=gpu:h100:1
#SBATCH --array=0-3

# Slavic languages we consider:
# be – Belarusian (East Slavic)
# bg – Bulgarian (South Slavic)
# bs – Bosnian (South Slavic)
# cs – Czech (West Slavic)
# hr – Croatian (South Slavic)
# mk – Macedonian (South Slavic)
# pl – Polish (West Slavic)
# ru – Russian (East Slavic)
# sk – Slovak (West Slavic)
# sl – Slovenian (South Slavic)
# sr – Serbian (South Slavic)
# uk – Ukrainian (East Slavic)

languages=("be" "bg" "bs" "cs" "hr" "mk" "pl" "ru" "sk" "sl" "sr" "uk")

echo "Launched at $(date)"
echo "Job ID: ${SLURM_JOBID}"
echo "Node list: ${SLURM_NODELIST}"
echo "Submit dir.: ${SLURM_SUBMIT_DIR}"
echo "Numb. of cores: ${SLURM_CPUS_PER_TASK}"

# Get total RAM in GB
total_ram_gb=$(awk '/MemTotal/ {printf "%.2f", $2 / 1024 / 1024}' /proc/meminfo)

# Get total VRAM in GB (Assuming NVIDIA GPU)
if command -v nvidia-smi &> /dev/null; then
    total_vram_gb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{printf "%.2f", $1 / 1024}')
else
    total_vram_gb="N/A"
fi

# Print results
echo "Total RAM: ${total_ram_gb} GB"
echo "Total VRAM: ${total_vram_gb} GB"

module load Anaconda3

echo "'conda run' for language ${languages[$SLURM_ARRAY_TASK_ID]}"

# Running the actual job
apptainer exec --nv --cleanenv \
    -B "$(pwd)":/workspace \
    vkovtun-tner-gpu.sif \
    conda run -n pytorch-env python3 /workspace/tner_fine_tuner_wikiann.py ${languages[$SLURM_ARRAY_TASK_ID]}