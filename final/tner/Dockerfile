# Dockerfile for tner==0.2.4 with GPU (CUDA 11.3 + PyTorch 1.12.1)

FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Fail early if CUDA is not enabled.
ENV USE_CUDA=1

ENV CONDA_ENV=pytorch-env

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
    echo "conda activate $CONDA_ENV" >> ~/.bashrc

ENV PATH="/opt/anaconda/bin:$PATH"

RUN nvcc -V

# Instrcutions are from here: https://github.com/pytorch/pytorch#from-source

RUN conda create -y -n $CONDA_ENV python=3.9 && \
    conda install -y -n $CONDA_ENV \
        pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 \
        cudatoolkit=11.3 -c pytorch -c nvidia && \
    conda install -n $CONDA_ENV -y "numpy<2" && \
    conda clean -ay

# Install tner and other dependencies
# Install scientific & NLP tools
RUN conda run -n $CONDA_ENV pip install \
    allennlp==2.10.1 \
    tner==0.2.4 \
    fairscale==0.4.6 \
    wandb \
    seqeval \
    sentencepiece \
    pydantic==1.7.4 \
    cached_path==1.1.6 \
    termcolor==1.1.0 \
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