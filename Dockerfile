FROM nvidia/cuda:11.7.1-base-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10-dev \
    python3-pip \
    libffi-dev \
    build-essential \
    rsync

# set the working directory 
WORKDIR /mnt

# copy the requirements.text file needed to install the package to the set working directory
COPY requirements.txt .

# upgrade pip
RUN python3 -m pip install --upgrade pip

# install the requirements
RUN pip3 install -r requirements.txt

# ensure right numba caching for umap-learn package
ENV NUMBA_CACHE_DIR=/tmp/numba_cache