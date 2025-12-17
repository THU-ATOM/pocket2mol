FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

SHELL ["/bin/bash", "-cu"]

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    g++-7 \
    git \
    curl \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ARG python=3.9
ENV PYTHON_VERSION=${python}
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/install-conda.sh \
    && bash /tmp/install-conda.sh -b -f -p /usr/local \
    && rm -f /tmp/install-conda.sh \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && conda install -y python=${PYTHON_VERSION} \
    && conda clean -ya

# Configure pip mirror
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Create conda environment
COPY ./docker/environment.yml /tmp/environment.yml
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && conda env create -f /tmp/environment.yml \
    && conda clean -ya

# Install PyTorch (with proxy support)
ARG http_proxy
ARG https_proxy
RUN conda run -n cbgbench bash -c "http_proxy=${http_proxy:-} https_proxy=${https_proxy:-} pip install torch==2.0.1 -i https://pypi.tuna.tsinghua.edu.cn/simple"
RUN conda run -n cbgbench bash -c "http_proxy=${http_proxy:-} https_proxy=${https_proxy:-} pip install torchvision==0.15.2 -i https://pypi.tuna.tsinghua.edu.cn/simple"
RUN conda run -n cbgbench bash -c "http_proxy=${http_proxy:-} https_proxy=${https_proxy:-} pip install torchaudio==2.0.2 -i https://pypi.tuna.tsinghua.edu.cn/simple"

# Fix NumPy version
RUN conda run -n cbgbench pip install "numpy>=1.24.0,<2.0"

# Install PyG extensions with CUDA support from PyG wheel server
# PyTorch 2.0.1 + CUDA 11.8 → use cu118
RUN conda run -n cbgbench bash -c "http_proxy=${http_proxy:-} https_proxy=${https_proxy:-} pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.1+cu118.html"
RUN conda run -n cbgbench bash -c "http_proxy=${http_proxy:-} https_proxy=${https_proxy:-} pip install torch-geometric -i https://pypi.tuna.tsinghua.edu.cn/simple"

# Configure pip mirror in cbgbench environment
RUN conda run -n cbgbench pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install chemistry packages
RUN conda run -n cbgbench pip install --use-pep517 EFGs biopython lxml \
    && conda run -n cbgbench conda install -y openbabel tensorboard tqdm pyyaml easydict python-lmdb rdkit -c conda-forge \
    && conda clean -ya

# Install docking tools
RUN conda run -n cbgbench conda install -y swig boost-cpp sphinx sphinx_rtd_theme -c conda-forge && conda clean -ya

ARG http_proxy
ARG https_proxy
RUN conda run -n cbgbench bash -c "http_proxy=${http_proxy:-} https_proxy=${https_proxy:-} python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3" || true

RUN conda run -n cbgbench pip install meeko==0.1.dev3 scipy pdb2pqr vina

# Force NumPy <2.0 after all conda installs
RUN conda run -n cbgbench pip install --force-reinstall "numpy>=1.24.0,<2.0"

# Install AIDD dependencies
RUN conda run -n cbgbench pip install hydra-core==1.3.0 hydra-colorlog==1.1.0 matplotlib pandas==1.5.3 tabulate==0.9.0 rich==13.4.2 biopython==1.79 docker

# Add user
ARG USER_ID=1031
ARG GROUP_ID=1033
ARG USER_NAME=docker
RUN if [ $USER_NAME != "root" ] ; then \
        addgroup --gid ${GROUP_ID} ${USER_NAME} && \
        adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID ${USER_NAME} && \
        usermod -aG sudo ${USER_NAME} && \
        echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers ; \
    fi

USER ${USER_NAME}
WORKDIR /workspace

# Initialize conda in user's bashrc
RUN echo "source /usr/local/etc/profile.d/conda.sh && conda activate cbgbench" >> ~/.bashrc

# Copy source code (THIS SHOULD BE LAST to maximize cache reuse)
COPY --chown=${USER_NAME}:${USER_NAME} . /workspace/
