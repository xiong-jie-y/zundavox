FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu20.04

# Prevent apt installation stop by EURO agreement.
ENV DEBIAN_FRONTEND teletype
ENV ACCEPT_EULA y

# To enable OPENGL window open in host side.
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

RUN apt-get update && apt-get install python3 python3-pip python3-dev python3-numpy -y
RUN apt-get install git-all -y

RUN pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# For matplotlib and simpleaudio
RUN apt-get install -y libfreetype6-dev libasound2-dev libfontconfig1-dev

RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html && \
    git clone https://github.com/xiong-jie-y/zundavox.git && \
    cd zundavox && \
    pip install -e . && \
    pip install git+https://github.com/xinntao/Real-ESRGAN && \
    pip install anime-face-detector

RUN apt-get install -y wget

RUN cd zundavox && mkdir -p data && \
    wget -O data/combiner.pt https://www.dropbox.com/s/at2r3v22xgyoxtk/combiner.pt?dl=0 && \
    wget -O data/eyebrow_decomposer.pt https://www.dropbox.com/s/pbomb5vgens03rk/eyebrow_decomposer.pt?dl=0 && \
    wget -O data/eyebrow_morphing_combiner.pt https://www.dropbox.com/s/yk9m5ok03e0ub1f/eyebrow_morphing_combiner.pt?dl=0 && \
    wget -O data/face_morpher.pt https://www.dropbox.com/s/77sza8qkiwd4qq5/face_morpher.pt?dl=0 && \
    wget -O data/two_algo_face_rotator.pt https://www.dropbox.com/s/ek261g9sspf0cqi/two_algo_face_rotator.pt?dl=0

RUN touch ~/.zundavox.yaml

RUN apt-get install p7zip-full -y

RUN wget https://github.com/VOICEVOX/voicevox_engine/releases/download/0.9.5/linux-nvidia.7z.001 && \
    wget https://github.com/VOICEVOX/voicevox_engine/releases/download/0.9.5/linux-nvidia.7z.002 && \
    wget https://github.com/VOICEVOX/voicevox_engine/releases/download/0.9.5/linux-nvidia.7z.txt && \
    7z x linux-nvidia.7z.001

RUN cd ./linux-nvidia && chmod +x ./run

RUN pip install numpy==1.21.0

# For librosa.
RUN apt-get install libsndfile1 -y

# for moviepy
RUN apt-get install ffmpeg -y

RUN cd zundavox && pip install -r requirements.txt

# This order and versions are important to make ginza works.
RUN pip install transformers==4.11.3 -U && \
    pip install ja-ginza-electra==5.0.0 -U && \
    pip install ginza-transformers==0.3.2 && \
    pip install SudachiTra==0.1.5 && \
    pip install huggingface-hub==0.0.19

WORKDIR /workspace/zundavox

RUN apt-get install -y gosu

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]