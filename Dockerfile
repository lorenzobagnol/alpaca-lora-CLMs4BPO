FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y wget
ENV DEBIAN_FRONTEND=

WORKDIR /root/
# Insall miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN chmod +x Miniconda3-latest-Linux-x86_64.sh
RUN /root/Miniconda3-latest-Linux-x86_64.sh -b
# Add conda to PATH
ENV PATH="/root/miniconda3/bin:${PATH}"
RUN conda init && conda clean -afy

# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

# Insall the right version of python and the required packages
RUN conda create --name alpaca python==3.10
WORKDIR /home/workspace/
# COPY . ./alpaca-CLMs4BPO
COPY requirements.txt ./alpaca-CLMs4BPO/requirements.txt
RUN conda run -n alpaca pip install --no-cache-dir -r ./alpaca-CLMs4BPO/requirements.txt