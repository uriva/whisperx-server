FROM python:3.11.6-bookworm
RUN apt-get -y update && apt-get -y upgrade
COPY requirements.txt ./
RUN pip install -r requirements.txt
RUN apt-get install -y ffmpeg && apt-get clean
# make whisperx download the model
RUN whisperx examples/sample01.wav --model small --align_model WAV2VEC2_ASR_LARGE_LV60K_960H --batch_size 4;  exit 0
COPY . .
CMD python src/server.py --torch_threads=1 --model_size=small --device=cuda