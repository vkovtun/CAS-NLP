#!/bin/bash
#SBATCH --job-name="Multi language named entities recognition. SpaCy training."
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=2GB
#SBATCH --cpus-per-task=4
#SBATCH --qos=job_gpu_preemptable
#SBATCH --gres=gpu:rtx4090:1

echo "Launched at $(date)"
echo "Job ID: ${SLURM_JOBID}"
echo "Node list: ${SLURM_NODELIST}"
echo "Submit dir.: ${SLURM_SUBMIT_DIR}"
echo "Numb. of cores: ${SLURM_CPUS_PER_TASK}"

echo "Setting export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Current directory:"
pwd

echo "module local Anaconda3"
module load Anaconda3
eval "$(conda shell.bash hook)"


#echo "conda create --name vegetation-anomalies"
conda create -y --name mnre

#echo "conda activate mnre"
conda activate mnre

echo "conda install -y -c conda-forge ..."
python -m cupyx.tools.install_library --cuda 12.x --library cutensor
conda install -y conda-forge::cupy conda-forge::spacy conda-forge::spacy-transformers conda-forge::thinc conda-forge::pytorch
#pip install 'spacy[transformers]' 'thinc[torch]'

#echo "SpaCy init."
#python -m spacy init config config_1.cfg --lang xx --pipeline ner --optimize accuracy --output config_filled.cfg

#echo "SpaCy debug."
#python -m spacy debug data config.cfg --paths.train ./train --paths.dev ./validation --paths.test ./test

echo "SpaCy train."
python -m spacy train config_1.cfg --output model_wikianc_en_1 --gpu-id 0


echo "conda deactivate"
conda deactivate
