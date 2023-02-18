export FLASK_APP=server
export FLASK_DEBUG=false
export WHISPER_MODEL=large-v2
export WHISPER_DEVICE=cpu
export TORCH_THREADS=4
python3.8 server.py
