<h1 align="center">WhisperX Server</h1>

This is a simple Server Application that receives audio file paths via the endpoint `POST transcribe`, it then uses Whisper ([WhisperX](https://github.com/m-bain/whisperX) to Transcribe and allign the transcription and outputs the result (`transcript.txt` and `subtitles.srt` files) to the directory of the input file, or to the directory that was indicated in the request.

For the server specification (request structure and response behavior) see the OpenAPI specificaiton in `swagger.yaml`.

This server will use the Whisper model size `large-v2`. To improve performance on account of accuracy, change the value of `WHISPER_MODEL` in `run.sh` to either of `["large", "large-v2", "medium", "small", "tiny"]

For any other documentation refer to [WhisperX readme](https://github.com/m-bain/whisperX).