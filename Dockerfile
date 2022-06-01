
FROM nvcr.io/nvidia/pytorch:22.04-py3

RUN DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Dubai \
    apt-get update && apt-get install -y 
ENV TZ=Asia/Dubai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /code

RUN apt-get update 
RUN apt-get install ffmpeg libsm6 libxext6  -y



RUN conda install -c conda-forge faiss-gpu
RUN (printf '#!/bin/bash\nunset TORCH_CUDA_ARCH_LIST\nexec \"$@\"\n' >> /entry.sh) && chmod a+x /entry.sh
ENTRYPOINT ["/entry.sh"]