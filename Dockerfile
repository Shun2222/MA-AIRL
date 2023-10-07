FROM tensorflow/tensorflow:1.14.0-gpu-py3 AS cuda-image
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

#FROM python:3.6 AS python-image

FROM cuda-image AS final-image
#COPY --from=python-image /usr/local /usr/local

RUN apt update
RUN apt -y upgrade
RUN apt-get -y install \
    clang \
    curl \
    git \
    vim \
    ffmpeg \
    sudo \
    cmake \
    git \
    libboost-system-dev \ 
    libjsoncpp-dev \
    libwebsocketpp-dev \
    mpich

ENV CODE_DIR /root/code/MA-AIRL

COPY . $CODE_DIR

WORKDIR $CODE_DIR
RUN python -m pip install -U pip

#RUN apt-get install -y libopenmpi-devcd apt --fix-broken install
#RUN apt install mpich
RUN pip install mpi4py

RUN pip install gym==0.10.5
RUN pip install numpy==1.16.5
RUN pip install click
RUN pip install scipy
RUN pip install pyglet==1.3.2
RUN pip install box2d
RUN pip install box2d-py
RUN pip install --upgrade pip setuptools wheel
RUN pip install pyproject-toml
RUN pip install matplotlib
RUN pip install wheel

WORKDIR $CODE_DIR/baselines
RUN pip install -e .

WORKDIR $CODE_DIR/multi-agent-particle-envs
RUN pip install -e .

WORKDIR $CODE_DIR/multi-agent-irl
RUN pip install -r requirements.txt

WORKDIR $CODE_DIR

