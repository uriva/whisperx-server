import logging
from typing import Dict

import modal

from src import worker

stub = modal.Stub("whisperx-service")


def load_model():
    import whisperx

    return whisperx.load_model("large-v3", "cuda")


logging.basicConfig(level=logging.INFO)


@stub.cls(
    image=(
        modal.Image.debian_slim()
        .apt_install("ffmpeg")
        .apt_install("git")
        .pip_install_from_requirements("./requirements.txt", gpu="any")
        .run_function(load_model, gpu="any")
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
