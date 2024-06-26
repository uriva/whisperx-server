import logging
from typing import Dict

import modal

from src import worker

app = modal.App("whisperx-service")

_model_size = "large-v3"


def load_model():
    import whisperx

    return whisperx.load_model(_model_size, "cuda")


logging.basicConfig(level=logging.INFO)

_example_video_file = "https://www2.cs.uic.edu/~i101/SoundFiles/taunt.wav"


@app.cls(
    image=(
        modal.Image.debian_slim(python_version="3.10")
        .apt_install("git")
        .pip_install(
            "git+https://github.com/openai/whisper.git",
            "dacite",
            "jiwer",
            "ffmpeg-python",
            "gql[all]~=3.0.0a5",
            "pandas",
            "loguru==0.6.0",
            "torchaudio==2.1.0",
        )
        .apt_install("ffmpeg")
        .pip_install("ffmpeg-python")
        .pip_install("gamla")
        .pip_install("git+https://github.com/m-bain/whisperx.git", gpu="any")
        .run_commands(
            f'whisperx --model {_model_size} "{_example_video_file}"', gpu="any"
        )
    ),
    gpu="any",
)
class Model:
    @modal.enter()
    def load_model(self):
        self._model = load_model()

    @modal.web_endpoint(method="POST")
    def predict(self, request: Dict):
        from whisperx import alignment

        language = request["language"]
        task = request["task"]
        audio_path = request["audio_path"]

        logging.info(f"{task} {audio_path} {language}")
        try:
            result = self._model.transcribe(audio_path, language=language, task=task)
            return worker.write_srt(
                alignment.align(
                    result["segments"],
                    *alignment.load_align_model(result["language"], self._model.device),
                    audio_path,
                    self._model.device,
                )["segments"]
            )
        except Exception as e:
            logging.error(e)
            return None
