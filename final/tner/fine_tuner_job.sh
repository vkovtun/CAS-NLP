#!/bin/bash
#SBATCH --job-name="TNER Fine Tuning"
#SBATCH --time=96:00:00
#SBATCH --mem-per-cpu=2GB
#SBATCH --cpus-per-task=40
#SBATCH --qos=job_cpu
#SBATCH --partition=epyc2
#_SBATCH --gres=gpu:rtx3090:1
#_SBATCH --gres=gpu:rtx4090:1
#_SBATCH --gres=gpu:a100:1
#_SBATCH --gres=gpu:h100:4
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
conda run -n tner python tner_fine_tuner_wikiann.py ${languages[$SLURM_ARRAY_TASK_ID]}
