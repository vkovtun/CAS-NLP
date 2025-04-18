# Dockerfile for tner==0.2.4 with GPU (CUDA 11.3 + PyTorch 1.12.1)
# Compatible with NVIDIA RTX GPUs, validated with CUDA 12.2 drivers

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install basic OS dependencies
RUN apt-get update && \
    apt-get install -y python3.9 python3.9-dev python3-pip git wget curl && \
    ln -sf python3.9 /usr/bin/python3 && \
    python3 -m pip install --upgrade pip setuptools wheel && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /workspace

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
