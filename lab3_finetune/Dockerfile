FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

# libaio-dev is required to build DeepSpeed NVMe (aio) offload support
RUN apt-get update && apt-get install -y \
    libaio-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "transformers>=4.47" \
    "trl>=0.13" \
    "peft>=0.14" \
    "accelerate>=1.2" \
    datasets \
    bitsandbytes \
    opencc-python-reimplemented

# DS_BUILD_AIO=1 compiles the async-IO op that enables NVMe offload
RUN DS_BUILD_AIO=1 pip install --no-cache-dir deepspeed

WORKDIR /workspace
