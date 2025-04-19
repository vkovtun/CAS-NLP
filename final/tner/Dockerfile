# Dockerfile for tner==0.2.4 with GPU (CUDA 11.3 + PyTorch 1.12.1)

FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Bash shell is needed
SHELL ["/bin/bash", "-c"]

# Install basic OS dependencies
RUN apt-get update && \
    apt-get install -y python3.9 python3.9-dev python3-pip git curl && \
    ln -sf python3.9 /usr/bin/python3 && \
    python3 -m pip install --upgrade pip setuptools wheel && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /workspace

# Install Anaconda for PyTorch manual installation
RUN curl https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh --output anaconda.sh && \
    bash anaconda.sh -b -p /opt/anaconda && \
    rm anaconda.sh && \
    ln -s /opt/anaconda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/anaconda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate pytorch-env" >> ~/.bashrc

ENV PATH="/opt/anaconda/bin:$PATH"

# Install PyTorch 1.12.1 with CUDA 11.3 directly from PyTorch
#RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 \
#    --extra-index-url https://download.pytorch.org/whl/cu113

# Instrcutions are from here: https://github.com/pytorch/pytorch#from-source

RUN conda create -y -n pytorch-env python=3.9 && \
    conda activate pytorch-env && \
    git clone --recursive --branch v1.12.1 https://github.com/pytorch/pytorch.git && \
    cd pytorch && \
    conda install -y cmake ninja numpy pyyaml mkl mkl-include setuptools cffi typing_extensions future six requests dataclasses && \
    pip install -r requirements.txt && \
    pip install mkl-static mkl-include && \
    conda install -y -c pytorch magma-cuda113 && \
    export CMAKE_PREFIX_PATH="${CONDA_PREFIX}" && \
    export TORCH_CUDA_ARCH_LIST="8.6" && \
    python setup.py install && \
    cd .. && rm -rf pytorch

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
