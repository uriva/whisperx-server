export FLASK_APP=server
export FLASK_DEBUG=false
export WHISPER_MODEL=large-v2
export WHISPER_DEVICE=cuda
python3.8 server.py
