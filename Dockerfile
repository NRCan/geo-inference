# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG WORKDIR=/usr/app
ARG USERNAME=gdl_user
ARG USERID=1000

# RNCAN certificate; uncomment (with right .cer name) if you are building behind a FW
# COPY NRCan-RootCA.cer /usr/local/share/ca-certificates/cert.crt
# RUN chmod 644 /usr/local/share/ca-certificates/cert.crt \
#     && update-ca-certificates 

# install required distribution packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends git unzip bzip2 build-essential python3-venv python-is-python3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# define a non root user to run the app
RUN useradd --create-home -s /bin/bash --no-user-group -u $USERID $USERNAME \
    && chown -R $USERNAME $WORKDIR

# switch to non root user
USER $USERNAME

# create python venv and install software
RUN python -m venv $WORKDIR/venv && $WORKDIR/venv/bin/pip install --upgrade pip
ENV PATH="$WORKDIR/venv/bin:$PATH"
RUN pip install --no-cache-dir git+https://github.com/NRCan/geo-inference.git \
    && pip uninstall -y pip
