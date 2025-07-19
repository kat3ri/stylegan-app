FROM rapidsai/rapidsai-core:cuda11.8-runtime-ubuntu22.04-py3.10

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    libboost-all-dev \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    git \
    curl \
    python3-dev \
    libffi-dev \
    libssl-dev \
    libjpeg-dev \
    zlib1g-dev

RUN pip install --upgrade pip

RUN pip install jupyterlab tqdm pillow ninja dlib \
  https://github.com/podgorskiy/dnnlib/releases/download/0.0.1/dnnlib-0.0.1-py3-none-any.whl

COPY . /workspace

WORKDIR /workspace

ENV PYTHONPATH=/workspace/src


EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
