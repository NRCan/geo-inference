# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu20.04

ARG CONDA_PYTHON_VERSION=3
ARG CONDA_DIR=/opt/conda
ARG USERNAME=gdl_user
ARG USERID=1000


# Install Mamba directly
ENV PATH=$CONDA_DIR/bin:$PATH

# RNCAN certificate; uncomment (with right .cer name) if you are building behind a FW
COPY NRCan-RootCA.cer /usr/local/share/ca-certificates/cert.crt
RUN chmod 644 /usr/local/share/ca-certificates/cert.crt \
    && update-ca-certificates \ 
    && apt-get update \
    && apt-get install -y --no-install-recommends git wget unzip bzip2 build-essential sudo \
    && apt-key del 7fa2af80 \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb \
    && sudo dpkg -i cuda-keyring_1.0-1_all.deb \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004-keyring.gpg \
    && sudo mv cuda-ubuntu2004-keyring.gpg /usr/share/keyrings/cuda-archive-keyring.gpg \
    && rm -f cuda-keyring_1.0-1_all.deb && rm -f /etc/apt/sources.list.d/cuda.list \
    && wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh -O /tmp/mamba.sh && \
    /bin/bash /tmp/mamba.sh -b -p $CONDA_DIR && \
    rm -rf /tmp/* && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=$CONDA_DIR/lib:$LD_LIBRARY_PATH

# Create the user
RUN useradd --create-home -s /bin/bash --no-user-group -u $USERID $USERNAME && \
    chown $USERNAME $CONDA_DIR -R && \
    adduser $USERNAME sudo && \
    echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

COPY requirements.txt .
USER $USERNAME
WORKDIR /home/$USERNAME/
COPY requirements.txt . /home/$USERNAME/
RUN cd /home/$USERNAME && \
    conda config --set ssl_verify no && \
    mamba create --name geo-inference && \ 
    mamba install pip && \
    pip install --upgrade pip && \
    pip install --no-cache-dir git+https://github.com/NRCan/geo-inference.git && \
    pip install --no-cache-dir -r /home/$USERNAME/requirements.txt && \
    pip uninstall -y pip && \
    mamba clean --all


ENV PATH=$CONDA_DIR/envs/geo-inference/bin:$PATH
RUN echo "source activate geo-inference" > ~/.bashrc