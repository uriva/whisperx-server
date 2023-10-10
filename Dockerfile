FROM python:3.11.6-bookworm
RUN git clone https://github.com/uriva/whisperx-server && \
    cd whisperx-server && \
    git checkout a2c01d7e122edbec520d580d022ff5e3f7b44110
RUN apt-get -y update && apt-get -y upgrade
RUN pip install -r whisperx-server/requirements.txt
RUN apt-get install -y ffmpeg && apt-get clean
# make whisperx download the model
RUN whisperx examples/sample01.wav --model small --align_model WAV2VEC2_ASR_LARGE_LV60K_960H --batch_size 4;  exit 0

COPY . .
CMD python whisperx-server/server.py --torch_threads=1 --model_size=small --device=cuda