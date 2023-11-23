import logging
from typing import Dict

import modal
from whisperx import load_model
from whisperx.alignment import align, load_align_model

from src import worker

stub = modal.Stub("whisperx-service")

_example_video_file = "https://www2.cs.uic.edu/~i101/SoundFiles/taunt.wav"


@stub.function(
    image=(
        modal.Image.debian_slim()
        .apt_install("ffmpeg")
        .apt_install("git")
        .pip_install_from_requirements("./requirements.txt")
        .run_commands(
            list(
                map(
                    lambda version: f'whisperx --model {version} "{_example_video_file}"',
                    ["tiny", "small", "medium", "large", "large-v2"],
                )
            ),
            gpu="any",
        )
    ),
    gpu="any",
)
@modal.web_endpoint(method="POST")
def _transcribe(request: Dict):
    language = request["language"]
    task = request["task"]
    audio_path = request["audio_path"]

    model = load_model(request["model"], "cuda")

    logging.info(f"{task} {audio_path} {model.device} {language}")
    try:
        result = model.transcribe(audio_path, language=language, task=task)
        align_model, align_metadata = load_align_model(result["language"], model.device)
        return worker.write_srt(
            align(
                result["segments"],
                align_model,
                align_metadata,
                audio_path,
                model.device,
            )["segments"]
        )
    except Exception as e:
        logging.error(e)
        return None
