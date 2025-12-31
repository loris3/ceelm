FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04 AS builder
RUN apt-get update 
RUN apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    git \
    cmake \
    python3-dev build-essential \
    && rm -rf /var/lib/apt/lists/*\
    # python3-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3.10 -m pip install --user --no-cache-dir --upgrade pip setuptools wheel

# RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126

COPY requirements.txt .
RUN VLLM_INSTALL_PUNICA_KERNELS=1 pip install --user --no-cache-dir --no-build-isolation -r requirements.txt
RUN git clone https://github.com/decile-team/submodlib.git && cd submodlib  && pip install --user --no-cache-dir --no-build-isolation . && cd ..


# limit number of jobs or will run out of RAM, see https://github.com/Dao-AILab/flash-attention
RUN MAX_JOBS=2 pip install --user --no-cache-dir --no-build-isolation flash-attn traker[fast] bitsandbytes 


# RUN git clone https://github.com/allenai/olmes.git && cd olmes  && pip install --user --no-cache-dir --no-build-isolation -e . && cd ..

RUN pip show datasets
# RUN pip install --user --no-cache-dir --no-build-isolation 
RUN pip uninstall -y pybind11 setuptools_scm

RUN rm -rf /app/submodlib 
RUN rm -rf /app/olmes 
RUN apt-get purge -y \
    build-essential \
    cmake \
    git \
    python3-dev \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


RUN rm -rf /usr/local/cuda/samples


RUN rm -rf ~/.cache /root/.cache /tmp/*

FROM nvidia/cuda:12.6.3-base-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update 
RUN apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    build-essential \
    rsync\
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local

RUN ln -s /usr/bin/python3.10 /usr/bin/python3


ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/root/.local/lib/python3.10/site-packages

ENV HF_HUB_ENABLE_HF_TRANSFER=1
WORKDIR /app

RUN ln -s /usr/bin/python3.10 /usr/bin/python


CMD ["bash"]