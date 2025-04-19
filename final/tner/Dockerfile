# Dockerfile for tner==0.2.4 with GPU (CUDA 11.3 + PyTorch 1.12.1)

FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

ENV TORCH_CUDA_ARCH_LIST="8.6"

# Install basic OS dependencies
RUN apt-get update && \
    apt-get install -y python3.9 python3.9-dev python3-pip git wget curl && \
    ln -sf python3.9 /usr/bin/python3 && \
    python3 -m pip install --upgrade pip setuptools wheel && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /workspace

# Install Conda for PyTorch manual installation
RUN curl https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh --output anaconda.sh && \
    echo "3ba0a298155c32fbfd80cbc238298560bf69a2df511783054adfc151b76d80d8 anaconda.sh" | sha256sum -c && \
    bash anaconda.sh -b -f -p ./anaconda && \
    rm anaconda.sh

# Install PyTorch 1.12.1 with CUDA 11.3 directly from PyTorch
#RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 \
#    --extra-index-url https://download.pytorch.org/whl/cu113

# Instrcutions are from here: https://github.com/pytorch/pytorch#from-source

RUN ["/bin/bash", "-c", "source ./anaconda/bin/activate && \
    conda create -y -n pytorch-env && \
    conda activate pytorch-env && \
    git clone --recursive --branch v1.12.1 https://github.com/pytorch/pytorch.git && \
    cd pytorch && \
    conda install -y cmake ninja && \
    pip install -r requirements.txt && \
    pip install mkl-static mkl-include && \
    conda install -y -c pytorch magma-cuda113 && \
    export CMAKE_PREFIX_PATH=\"${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}\" && \
    python setup.py install && \
    conda deactivate && \
    conda remove -y -n pytorch-env --all"]

# Install tner and other dependencies
RUN pip install \
    tner==0.2.4 \
    allennlp==2.9.3 \
    fairscale==0.4.6 \
    "wandb<0.13.0" \
    seqeval \
    sentencepiece \
    pydantic==1.7.4 \
    cached_path==1.1.6 \
    termcolor==1.1.0

# Install scientific & NLP tools
RUN pip install \
    numpy==1.21.6 \
    pandas==1.3.5 \
    scikit-learn \
    scipy \
    nltk \
    h5py \
    spacy==3.2.4 \
    transformers==4.17.0 \
    tokenizers==0.11.6 \
    datasets==2.10.1 \
    tensorboardX \
    flask \
    "pyarrow<=12.0.1"
