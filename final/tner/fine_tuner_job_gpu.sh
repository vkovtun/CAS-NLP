#!/bin/bash
#SBATCH --job-name="TNER Fine Tuning"
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=32GB
#SBATCH --cpus-per-task=2

#_SBATCH --qos=job_gpu_preemptable
#_SBATCH --partition=gpu-invest

#_SBATCH --qos=job_gpu
#SBATCH --partition=gpu

#_SBATCH --gres=gpu:rtx3090:1
#_SBATCH --gres=gpu:rtx4090:1
#SBATCH --gres=gpu:a100:1
#_SBATCH --gres=gpu:h100:1
#SBATCH --array=0-2

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

echo "'conda run' for language ${languages[$SLURM_ARRAY_TASK_ID]}"

# Running the actual job
apptainer exec --nv --cleanenv \
    -B "$(pwd)":/workspace \
    vkovtun-tner-gpu.sif \
    conda run -n pytorch-env python3 /workspace/tner_fine_tuner_wikiann.py ${languages[$SLURM_ARRAY_TASK_ID]}