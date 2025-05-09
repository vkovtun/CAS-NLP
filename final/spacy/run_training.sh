#!/bin/bash
#SBATCH --job-name="Named entities recognition for Slavic Languages. SpaCy training."
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=64GB
#SBATCH --cpus-per-task=2
#SBATCH --qos=job_gpu_preemptable

#_SBATCH --gres=gpu:rtx3090:1
#_SBATCH --gres=gpu:rtx4090:1
#SBATCH --gres=gpu:a100:1
#_SBATCH --gres=gpu:h100:1
#SBATCH --array=0-11

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

echo "Setting export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "module local Anaconda3"
module load Anaconda3
eval "$(conda shell.bash hook)"

echo "conda activate mnre"
conda activate mnre

echo "SpaCy train for language ${languages[$SLURM_ARRAY_TASK_ID]}."
python -m spacy train config/wikianc/${languages[$SLURM_ARRAY_TASK_ID]}.cfg --output models/wikianc/${languages[$SLURM_ARRAY_TASK_ID]} --gpu-id 0

echo "conda deactivate"
conda deactivate
