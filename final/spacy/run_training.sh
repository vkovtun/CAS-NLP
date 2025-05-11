#!/bin/bash
#SBATCH --job-name="Named entities recognition for Slavic Languages. SpaCy training."
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=10GB
#SBATCH --cpus-per-task=4
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

language=${languages[$SLURM_ARRAY_TASK_ID]}
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

# Get max index from folders
max_index=$(find datasets/wikianc/${language}/ -maxdepth 1 -type d -regex ".*/[0-9]+" | sed 's#.*/##' | sort -n | tail -n1)

for index in $(seq 1 $max_index); do
    echo "Training on ${language}, split $index"

    cfg_file="config/wikianc/${language}.cfg"
    out_dir="models/wikianc/${language}"

    if [ -d "$out_dir" ]; then
        prev_model_dir="models/wikianc/${language}_prev"
        extra_params="--paths.pretrained ${prev_model_dir}/model-last"

        mv ${out_dir} ${prev_model_dir}
    else
        extra_params="--paths.ner_factory=ner --paths.transformer_factory=transformer"
    fi

    python -m spacy train "$cfg_file" \
      --output "$out_dir" \
      --gpu-id 0 \
      -VV \
      --paths.train "datasets/wikianc/${language}/${index}/train/" \
      --paths.dev   "datasets/wikianc/${language}/${index}/dev/" \
      ${extra_params}

done

echo "conda deactivate"
conda deactivate